
# Resolve companion modules when this historical script is invoked directly.
import sys as _sys
from pathlib import Path as _Path
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
_research_root = next(p for p in _Path(__file__).resolve().parents if p.name == "research_scripts")
_sys.path.append(str(_research_root))
_sys.path.append(str(_research_root.parent))
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from neuralop.models import FNO
from tqdm import tqdm

# ==========================================
# ⚙️ 1. 全局配置 (Configuration)
# ==========================================
DATA_ROOT = "export_images" 
RESULT_DIR = "training_results"

# --- 显存控制核心 ---
TRAIN_SIZE = 128       # ⚡ 训练分辨率 (建议 128/256 以节省显存)
TARGET_SIZE = 512      # 🎯 原始数据分辨率

BATCH_SIZE = 4         
EPOCHS = 200           
LEARNING_RATE = 1e-3   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

print(f"\n" + "="*40)
print(f"🤖 FNO 训练任务启动")
print(f"========================================")
print(f"🚀 运行设备: {DEVICE}")
print(f"📏 训练尺寸: {TRAIN_SIZE}x{TRAIN_SIZE} (自动适配 512 原图)")
print(f"💾 结果保存: {RESULT_DIR}")
print(f"="*40 + "\n")

# ==========================================
# 📂 2. 数据集加载器 (Dataset)
# ==========================================
class Battery512Dataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size
        
        self.sample_folders = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        
        if len(self.sample_folders) == 0:
            raise ValueError(f"❌ 错误: 在 {root_dir} 下没找到任何文件夹！")
            
        print(f"📦 数据集就绪: 发现 {len(self.sample_folders)} 个样本文件夹")

    def _load_img(self, path):
        if not os.path.exists(path):
            print(f"⚠️  警告: 缺失文件 {path} (将使用黑图代替)")
            raise ValueError(f"Missing or unreadable source image: {path}")
        
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (self.image_size, self.image_size)) 
        return img.astype(np.float32) / 255.0 

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):
        folder = self.sample_folders[idx]
        
        # --- A. 加载静态图 ---
        geo_path = os.path.join(folder, "3_Voronoi_Geometry", "05_Voronoi_Theta_Colored.png")
        ori_img = self._load_img(geo_path)

        crate_files = glob.glob(os.path.join(folder, "C-rate", "*.png"))
        if len(crate_files) > 0:
            crate_img = self._load_img(crate_files[0])
        else:
            raise ValueError(f"Missing C-rate image: {folder}")

        # --- B. 加载动态序列 ---
        conc_files = sorted(glob.glob(os.path.join(folder, "1_Concentration", "*.png")))
        stress_files = sorted(glob.glob(os.path.join(folder, "2_Stress", "*.png")))
        
        SEQ_LEN = 25
        conc_list = []
        stress_list = []
        
        for i in range(SEQ_LEN):
            c_p = conc_files[i] if i < len(conc_files) else "missing"
            conc_list.append(self._load_img(c_p))
            
            s_p = stress_files[i] if i < len(stress_files) else "missing"
            stress_list.append(self._load_img(s_p))

        conc_stack_np = np.stack(conc_list, axis=0)
        stress_stack_np = np.stack(stress_list, axis=0)
        
        # --- C. 转 Tensor ---
        ori_t = torch.tensor(ori_img).permute(2, 0, 1)
        crate_t = torch.tensor(crate_img).permute(2, 0, 1)
        
        conc_t = torch.tensor(conc_stack_np).permute(0, 3, 1, 2).reshape(-1, self.image_size, self.image_size)
        stress_t = torch.tensor(stress_stack_np).permute(0, 3, 1, 2).reshape(-1, self.image_size, self.image_size)
        
        input_tensor = torch.cat([ori_t, crate_t, conc_t], dim=0) 
        target_tensor = stress_t 
        
        return input_tensor, target_tensor

# ==========================================
# 🎨 3. 可视化函数
# ==========================================
def save_visual_check(pred_tensor, target_tensor, epoch, save_dir):
    pred_np = pred_tensor[0].detach().cpu().numpy()
    target_np = target_tensor[0].detach().cpu().numpy()
    
    frames_idx = [0, 5, 11, 17, 24]
    
    plt.figure(figsize=(15, 6))
    
    for i, t in enumerate(frames_idx):
        start = t * 3
        end = start + 3
        
        p_img = pred_np[start:end, :, :].transpose(1, 2, 0)
        t_img = target_np[start:end, :, :].transpose(1, 2, 0)
        
        p_img = (p_img - p_img.min()) / (p_img.max() - p_img.min() + 1e-6)
        t_img = (t_img - t_img.min()) / (t_img.max() - t_img.min() + 1e-6)
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(p_img)
        plt.title(f"🤖 Pred: T={t}")
        plt.axis('off')
        
        plt.subplot(2, 5, i + 6)
        plt.imshow(t_img)
        plt.title(f"👁️ True: T={t}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_visual.png"))
    plt.close()
    print(f"🖼️  [可视化] Epoch {epoch} 对比图已生成 -> {save_dir}")

# ==========================================
# 🧠 4. 主训练程序
# ==========================================
if __name__ == "__main__":
    
    # 1. 准备数据
    try:
        full_dataset = Battery512Dataset(DATA_ROOT, image_size=TRAIN_SIZE)
    except Exception as e:
        print(f"❌ 致命错误: {e}")
        exit()
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"📊 数据划分: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)}")

    # 2. 构建模型
    print("🏗️  正在搭建 FNO 神经网络架构 (In=81 -> Out=75)...")
    model = FNO(
        n_modes=(16, 16),      
        hidden_channels=64,    
        in_channels=81,        
        out_channels=75        
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    # 3. 开始训练
    print("\n" + "🔥"*10 + " 开始训练 " + "🔥"*10 + "\n")
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # 带有 Emoji 的进度条
        loop = tqdm(train_loader, desc=f"🏃 Ep {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # 4. 验证与可视化
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                    pred_val = model(x_val)
                    val_loss += criterion(pred_val, y_val).item()
                
                # 画图检查
                save_visual_check(pred_val, y_val, epoch+1, RESULT_DIR)

            avg_val_loss = val_loss / len(val_loader)
            print(f"✨ Epoch {epoch+1} 完成 | Val Loss: {avg_val_loss:.6f}")
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "best_fno_512_model.pth")
                print(f"💾 [新纪录] 最佳模型已保存! Loss: {best_loss:.6f}")

    print("\n" + "🎉"*10 + " 训练全部完成! " + "🎉"*10)
    print(f"🏆 最终最佳 Val Loss: {best_loss:.6f}")
    print(f"📂 请查看 {RESULT_DIR} 获取可视化结果")