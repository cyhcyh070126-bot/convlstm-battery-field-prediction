import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import os
import random
import glob
import cv2
import re
from pytorch_msssim import ssim 
from convlstm import ConvLSTM

# =============================================================================
# --- ⚙️ [配置] 全局设置 ---
# =============================================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🖥️ 硬件配置: RTX 5090 (32GB) + 25 Core CPU
BATCH_SIZE = 64        
LEARNING_RATE = 1e-3   
EPOCHS = 100
INPUT_SEQ_LENGTH = 5   
PATCH_SIZE = 128       

VAL_INTERVAL = 5       # 每 5 个 Epoch 验证一次

# =============================================================================
# --- 1. 📂 数据集 (RAM 高速模式) ---
# =============================================================================
class RAMBatteryDataset(Dataset):
    def __init__(self, folder_list, is_training=True):
        self.samples_meta = []
        self.data_cache = []
        self.is_training = is_training
        
        self.patch_size = PATCH_SIZE
        self.patches_per_row = 512 // self.patch_size
        self.patches_per_img = self.patches_per_row ** 2
        
        mode_str = "🏋️ 训练集" if is_training else "🧪 验证集"
        print(f"🚀 [RAM模式] 正在加载 {mode_str} ...")
        
        for folder in folder_list:
            conc_dir = os.path.join(folder, "1_Concentration")
            if not os.path.exists(conc_dir): continue
            
            # 按文件名数字排序
            files = glob.glob(os.path.join(conc_dir, "*.png"))
            files = sorted(files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
            
            num_frames = len(files)
            if num_frames < INPUT_SEQ_LENGTH + 1: continue

            try:
                ori_path = glob.glob(os.path.join(folder, "3_Voronoi_Geometry", "05_*.png"))[0]
                crate_path = glob.glob(os.path.join(folder, "C-rate", "*.png"))[0]
            except IndexError:
                continue
            
            # 读取静态图 & 动态图到内存
            static_imgs = (self._load_img(ori_path), self._load_img(crate_path))
            conc_imgs_ram = [self._load_img(f) for f in files]

            self.data_cache.append({
                'static': static_imgs,
                'conc': conc_imgs_ram
            })
            
            current_cache_idx = len(self.data_cache) - 1

            # 生成索引元数据
            for t_start in range(num_frames - INPUT_SEQ_LENGTH):
                for patch_id in range(self.patches_per_img):
                    self.samples_meta.append({
                        'cache_idx': current_cache_idx,
                        't_start': t_start,
                        'patch_id': patch_id
                    })
        
        print(f"  ✅ {mode_str} 就绪 | 📁 文件夹: {len(self.data_cache)} | 🔢 样本切片: {len(self.samples_meta)}")
    
    def _load_img(self, path):
        img = cv2.imread(path)
        if img is None: return np.zeros((512, 512, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512)) 
        return img.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples_meta)
    
    def __getitem__(self, idx):
        meta = self.samples_meta[idx]
        cache_item = self.data_cache[meta['cache_idx']]
        ori_map, crate_map = cache_item['static']
        
        # 获取时间序列
        seq_imgs = cache_item['conc'][meta['t_start'] : meta['t_start'] + INPUT_SEQ_LENGTH + 1]
        
        # 堆叠通道
        frames_list = []
        for conc_img in seq_imgs:
            frame_stack = np.concatenate([conc_img, ori_map, crate_map], axis=-1)
            frames_list.append(frame_stack)
            
        full_seq_np = np.stack(frames_list) # (T, H, W, C)
        
        # 空间切块
        row = meta['patch_id'] // self.patches_per_row
        col = meta['patch_id'] % self.patches_per_row
        h_s, w_s = row * self.patch_size, col * self.patch_size
        patch_np = full_seq_np[:, h_s:h_s+self.patch_size, w_s:w_s+self.patch_size, :]
        
        # 转 Tensor: (T, H, W, C) -> (T, C, H, W)
        data_tensor = torch.from_numpy(patch_np).permute(0, 3, 1, 2).float()
        
        # 🎲 在线数据增强
        if self.is_training:
            if random.random() > 0.5: data_tensor = TF.hflip(data_tensor)
            if random.random() > 0.5: data_tensor = TF.vflip(data_tensor)
            if random.random() > 0.5: data_tensor = torch.rot90(data_tensor, 1, [2, 3])

        input_tensor = data_tensor[:INPUT_SEQ_LENGTH]       
        target_tensor = data_tensor[INPUT_SEQ_LENGTH, 0:3]  

        return input_tensor, target_tensor

# =============================================================================
# --- 2. 🧠 模型架构 [Efficient Forward] ---
# =============================================================================
class PaperModel(nn.Module):
    def __init__(self, input_channels=9, hidden_dim=32, kernel_size=(5, 5), num_layers=3):
        super(PaperModel, self).__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels=input_channels,
            out_channels=hidden_dim, 
            kernel_size=(3, 5, 5), 
            padding=(1, 2, 2) 
        )
        
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_dim,  
            hidden_dim=[hidden_dim] * num_layers,
            kernel_size=[kernel_size] * num_layers,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False 
        )
        
        self.final_conv = nn.Conv2d(
            in_channels=hidden_dim, 
            out_channels=3, 
            kernel_size=1, 
            padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 3D Conv
        x_3d = x.permute(0, 2, 1, 3, 4) 
        feature_3d = self.conv3d(x_3d)
        
        # 2. ConvLSTM
        lstm_input = feature_3d.permute(0, 2, 1, 3, 4)
        layer_output_list, _ = self.conv_lstm(lstm_input)
        all_time_steps = layer_output_list[0] 

        # 3. [⚡ 优化点] 只取最后一帧
        last_frame_feature = all_time_steps[:, -1, :, :, :]

        # 4. Decoding
        predicted = self.final_conv(last_frame_feature)
        predicted = self.sigmoid(predicted)
        
        return predicted

# =============================================================================
# --- 3. 📉 Loss & Main ---
# =============================================================================
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)
        try:
            ssim_val = ssim(preds, targets, data_range=1.0, size_average=True)
        except Exception:
            ssim_val = torch.tensor(0.0).to(preds.device)
        return mse_loss + self.alpha * (1.0 - ssim_val)

if __name__ == "__main__":
    print(f"\n✨ --- 启动训练程序 ---")
    print(f"🖥️  计算设备: {DEVICE} (RTX 5090 Ready)")
    print(f"🔥  CPU核心利用: 20 Threads")
    
    base_data_dir = "export_images"
    if not os.path.exists(base_data_dir):
        print(f"❌ 错误：找不到目录 {base_data_dir}")
        exit()

    all_folders = [os.path.join(base_data_dir, f) for f in os.listdir(base_data_dir) 
                   if os.path.isdir(os.path.join(base_data_dir, f))]
    
    if not all_folders:
        print("❌ 目录为空，无数据！")
        exit()

    random.seed(42) 
    random.shuffle(all_folders)

    train_split = int(len(all_folders) * 0.80)
    
    # === 1. 初始化数据集 ===
    train_dataset = RAMBatteryDataset(all_folders[:train_split], is_training=True)
    val_dataset = RAMBatteryDataset(all_folders[train_split:], is_training=False)

    # === 2. DataLoader (多核优化) ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=20,         # ⚡ 20线程并发加载
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    ) 
    
    # === 3. 模型与优化器 ===
    model = PaperModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = HybridLoss().to(DEVICE)
    
    print(f"\n🚂 --- 开始训练 (Epochs: {EPOCHS}) ---")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # --- 训练循环 ---
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            output_last_frame = model(inputs) 
            
            loss = criterion(output_last_frame, targets)
            loss.backward()
            
            # ✂️ [已确认] 梯度裁剪已移除
            # optimizer.step() 前无 clip_grad_norm_

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证循环 ---
        val_str = ""
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            model.eval() 
            val_loss = 0
            with torch.no_grad(): 
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    output_last_frame = model(inputs)
                    loss = criterion(output_last_frame, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_str = f"| 🧪 Val Loss: {avg_val_loss:.6f}"
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model_pytorch.pth")
                # 🇨🇳 修正为中文
                val_str += " (🏆 最佳模型!)"
        else:
            val_str = "| 💤 Val Skipped"

        print(f"Epoch {epoch+1}/{EPOCHS} | 📉 Train Loss: {avg_train_loss:.6f} {val_str}")

    print("\n🎉🎉 训练全部完成！模型已保存。")