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

# --- 防止 OpenCV 多线程冲突 ---
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# =============================================================================
# --- ⚙️ [配置] 全局参数 ---
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [显存与Batch配置]
# 训练用小图(128x128)，Batch可以大
TRAIN_BATCH_SIZE = 16    
# 验证用全图(512x512)，面积是训练的16倍，Batch必须极小！
VAL_BATCH_SIZE = 4       

LEARNING_RATE = 1e-5    
EPOCHS = 50             

# [序列设置]
INPUT_SEQ_LENGTH = 5    
SEQUENCE_PREDICT_LEN = 10 
TOTAL_SEQ_LEN = INPUT_SEQ_LENGTH + SEQUENCE_PREDICT_LEN 

PATCH_SIZE = 128  # 训练时的切片大小       

# [计划采样]
EPSILON_START = 1.0         
EPSILON_DECAY = 0.00001 
global_iteration_step = 0   

# =============================================================================
# --- 1. 📂 数据集 (全覆盖切片 + 全图验证) ---
# =============================================================================
class LazySeqBatteryDataset(Dataset):
    def __init__(self, folder_list, is_training=True, patch_size=128):
        self.samples_meta = []   
        self.is_training = is_training
        self.patch_size = patch_size
        self.static_cache = {} 
        
        # 计算切片逻辑
        self.patches_per_row = 512 // self.patch_size # 4
        self.patches_per_img = self.patches_per_row ** 2 # 16
        
        mode = "🏋️ 微调训练集" if is_training else "🧪 微调验证集"
        print(f"正在构建 {mode} 索引...")
        
        for folder in folder_list:
            conc_dir = os.path.join(folder, "1_Concentration")
            if not os.path.exists(conc_dir): continue
            
            files = glob.glob(os.path.join(conc_dir, "*.png"))
            files = sorted(files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
            
            num_frames = len(files)
            if num_frames < TOTAL_SEQ_LEN: continue

            # 缓存静态图
            try:
                ori_path = glob.glob(os.path.join(folder, "3_Voronoi_Geometry", "05_*.png"))[0]
                crate_path = glob.glob(os.path.join(folder, "C-rate", "*.png"))[0]
                if folder not in self.static_cache:
                    self.static_cache[folder] = (self._load_img(ori_path), self._load_img(crate_path))
            except IndexError:
                continue

            step = 1
            for t_start in range(0, num_frames - TOTAL_SEQ_LEN + 1, step):
                seq_files = files[t_start : t_start + TOTAL_SEQ_LEN]
                
                if self.is_training:
                    # --- 训练模式：生成 16 个切片的索引 ---
                    # 样本量膨胀 16 倍，保证全覆盖
                    for p_id in range(self.patches_per_img):
                        self.samples_meta.append({
                            'folder': folder,
                            'conc_paths': seq_files,
                            'patch_id': p_id  # 记录要切哪一块 (0-15)
                        })
                else:
                    # --- 验证模式：只生成 1 个索引 (全图) ---
                    self.samples_meta.append({
                        'folder': folder,
                        'conc_paths': seq_files,
                        'patch_id': -1    # -1 代表不切，用全图
                    })
        
        print(f"  ✅ {mode} 准备就绪 | 样本数: {len(self.samples_meta)}")
    
    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        return img.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples_meta)
    
    def __getitem__(self, idx):
        meta = self.samples_meta[idx]
        folder = meta['folder']
        conc_paths = meta['conc_paths']
        patch_id = meta['patch_id']
        
        ori_map, crate_map = self.static_cache[folder]
        
        frames_list = []
        for p in conc_paths:
            conc_img = self._load_img(p)
            frame_stack = np.concatenate([conc_img, ori_map, crate_map], axis=-1)
            frames_list.append(frame_stack)
            
        full_seq_np = np.stack(frames_list) # (15, 512, 512, 9)
        
        # --- 核心逻辑分支 ---
        if self.is_training and patch_id != -1:
            # === 训练模式：固定网格切片 ===
            row = patch_id // self.patches_per_row # 0-3
            col = patch_id % self.patches_per_row  # 0-3
            h_s, w_s = row * self.patch_size, col * self.patch_size
            
            # 精确切出 128x128
            patch_np = full_seq_np[:, h_s:h_s+self.patch_size, w_s:w_s+self.patch_size, :]
            full_seq_tensor = torch.from_numpy(patch_np).permute(0, 3, 1, 2).float()
            
            # 🎲 训练时加入翻转增强 (数据量够大，增强能防止过拟合)
            if random.random() > 0.5: full_seq_tensor = TF.hflip(full_seq_tensor)
            if random.random() > 0.5: full_seq_tensor = TF.vflip(full_seq_tensor)
            if random.random() > 0.5: full_seq_tensor = torch.rot90(full_seq_tensor, 1, [2, 3])
            
        else:
            # === 验证模式：使用全图 (512x512) ===
            # 直接转 Tensor，不切片，不翻转
            full_seq_tensor = torch.from_numpy(full_seq_np).permute(0, 3, 1, 2).float()

        input_tensor = full_seq_tensor[:INPUT_SEQ_LENGTH] 
        target_seq_tensor = full_seq_tensor[INPUT_SEQ_LENGTH:, 0:3, :, :] 

        return input_tensor, target_seq_tensor

# =============================================================================
# --- 2. 🧠 模型架构 ---
# =============================================================================
class PaperModel(nn.Module):
    def __init__(self, input_channels=9, hidden_dim=32, kernel_size=(5, 5), num_layers=3):
        super(PaperModel, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.conv_lstm = ConvLSTM(input_dim=hidden_dim, hidden_dim=[hidden_dim]*num_layers, kernel_size=[kernel_size]*num_layers, num_layers=num_layers, batch_first=True, return_all_layers=False)
        self.final_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=3, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_3d = x.permute(0, 2, 1, 3, 4) 
        feature_3d = self.conv3d(x_3d)
        lstm_input = feature_3d.permute(0, 2, 1, 3, 4)
        layer_output_list, _ = self.conv_lstm(lstm_input)
        all_time_steps = layer_output_list[0] 
        last_frame_feature = all_time_steps[:, -1, :, :, :]
        predicted = self.final_conv(last_frame_feature)
        predicted = self.sigmoid(predicted)
        return predicted

# =============================================================================
# --- 3. Loss & Main ---
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
    torch.cuda.empty_cache()
    print(f"--- 设备: {DEVICE} (RTX 5090 Sequence Finetuning - Full Validation) ---")
    
    base_data_dir = "export_images"
    if not os.path.exists(base_data_dir): 
        print("❌ 找不到数据目录"); exit()

    all_folders = [os.path.join(base_data_dir, f) for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    random.seed(42) 
    random.shuffle(all_folders)

    train_split = int(len(all_folders) * 0.90)
    
    # 训练集: 使用 patch_size=128 (会被切成16块)
    train_dataset = LazySeqBatteryDataset(all_folders[:train_split], is_training=True, patch_size=PATCH_SIZE)
    # 验证集: 使用 patch_size=128 (但代码里 logic 会忽略它，直接返回全图)
    val_dataset = LazySeqBatteryDataset(all_folders[train_split:], is_training=False, patch_size=PATCH_SIZE)

    # 🚀 训练 Loader: Batch=16, 20线程
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True, 
        num_workers=20, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # 🧪 验证 Loader: Batch=4 (因为图变大了16倍，必须减小Batch防止爆显存)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=VAL_BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    ) 
    
    model = PaperModel().to(DEVICE)
    
    if os.path.exists("best_model_pytorch.pth"):
        print("\n🔄 正在加载预训练权重...")
        try:
            model.load_state_dict(torch.load("best_model_pytorch.pth", map_location=DEVICE))
            print("✅ 权重加载成功！")
        except Exception as e:
            print(f"❌ 加载失败: {e}"); exit()
    else:
        print("⚠️ 未找到 best_model_pytorch.pth"); exit()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = HybridLoss().to(DEVICE)

    print(f"\n🚀 开始微调: {INPUT_SEQ_LENGTH}帧 -> 预测 {SEQUENCE_PREDICT_LEN}帧")
    print(f"📦 训练配置: 128x128 切片 (Batch {TRAIN_BATCH_SIZE})")
    print(f"📦 验证配置: 512x512 全图 (Batch {VAL_BATCH_SIZE})")
    
    best_val_loss = float('inf') 
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target_sequence) in enumerate(train_loader):
            epsilon = max(0, EPSILON_START - global_iteration_step * EPSILON_DECAY)
            global_iteration_step += 1
            
            current_input_seq = data.to(DEVICE) 
            batch_static_info = current_input_seq[:, -1, 3:9, :, :] 
            
            total_sequence_loss = 0
            optimizer.zero_grad()
            
            # 训练循环 (128x128)
            for t in range(SEQUENCE_PREDICT_LEN):
                prediction_t = model(current_input_seq)
                
                target_t = target_sequence[:, t].to(DEVICE)
                loss = criterion(prediction_t, target_t)
                total_sequence_loss += loss
                
                predicted_frame_9ch = torch.cat([prediction_t, batch_static_info], dim=1)
                
                next_frame_input = None
                if random.random() < epsilon:
                    if t + 1 < SEQUENCE_PREDICT_LEN: 
                        gt_conc_t = target_sequence[:, t].to(DEVICE)
                        gt_frame_9ch = torch.cat([gt_conc_t, batch_static_info], dim=1)
                        next_frame_input = gt_frame_9ch.unsqueeze(1)
                    else:
                        next_frame_input = predicted_frame_9ch.unsqueeze(1)
                else:
                    next_frame_input = predicted_frame_9ch.unsqueeze(1) 
                
                current_input_seq = torch.cat([current_input_seq[:, 1:, ...], next_frame_input], dim=1)
            
            avg_batch_loss = total_sequence_loss / SEQUENCE_PREDICT_LEN

            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += avg_batch_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  [Ep {epoch}] Step {batch_idx} Loss: {avg_batch_loss.item():.6f} (Eps: {epsilon:.2f})")
        
        avg_train_loss = train_loss / len(train_loader)

        # 🧪 验证循环 (512x512 全图验证)
        model.eval() 
        val_loss = 0
        with torch.no_grad(): 
            for data, target_sequence in val_loader:
                current_input = data.to(DEVICE) 
                # data shape: (4, 5, 9, 512, 512)
                
                batch_static = current_input[:, -1, 3:9, :, :]
                seq_loss = 0
                for t in range(SEQUENCE_PREDICT_LEN):
                    pred_t = model(current_input)
                    # pred_t shape: (4, 3, 512, 512) -> 只要显存够，Conv网络自适应大小
                    
                    target_t = target_sequence[:, t].to(DEVICE)
                    seq_loss += criterion(pred_t, target_t)
                    
                    pred_9ch = torch.cat([pred_t, batch_static], dim=1)
                    current_input = torch.cat([current_input[:, 1:], pred_9ch.unsqueeze(1)], dim=1)
                val_loss += (seq_loss / SEQUENCE_PREDICT_LEN).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Ep {epoch+1} | Train (128px): {avg_train_loss:.5f} | Val (512px): {avg_val_loss:.5f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_finetuned.pth")
            print("  --> 🌟 全图验证最佳模型已保存！")
            
    print("\n🎉 微调完成！")