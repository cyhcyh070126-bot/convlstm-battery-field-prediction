
# Resolve companion modules when this historical script is invoked directly.
import sys as _sys
from pathlib import Path as _Path
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
_research_root = next(p for p in _Path(__file__).resolve().parents if p.name == "research_scripts")
_sys.path.append(str(_research_root))
_sys.path.append(str(_research_root.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# 这里的 ssim 我们稍后在云端安装 'pytorch_msssim' 库即可使用
from pytorch_msssim import ssim 

# 导入你已有的模块
from data_loader import load_simulation_data
from convlstm import ConvLSTM

# =============================================================================
# --- 1. 配置参数 (与论文一致) ---
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1         #
LEARNING_RATE = 1e-3   #
EPOCHS = 50            #
INPUT_SEQ_LENGTH = 5   # 输入前5帧

# =============================================================================
# --- 2. 数据集包装器 (PyTorch Dataset) ---
# =============================================================================
class BatteryDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        
        all_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, f))]
        
        print(f"正在扫描数据... 发现 {len(all_folders)} 个样本文件夹。")
        
        for folder in all_folders:
            # 使用你的 data_loader 加载数据
            # 返回: (conc_seq, stress_seq)
            conc_seq, _ = load_simulation_data(folder)
            
            if conc_seq is not None:
                # 切片: 把 25 帧切成多个 (输入5帧 -> 输出1帧) 的样本
                total_frames = conc_seq.shape[0]
                for i in range(total_frames - INPUT_SEQ_LENGTH):
                    # 输入: 5帧, 9通道
                    input_seq = conc_seq[i : i + INPUT_SEQ_LENGTH]
                    # 目标: 第6帧, 3通道 (只预测浓度场)
                    target_frame = conc_seq[i + INPUT_SEQ_LENGTH][:,:,0:3] 
                    self.samples.append((input_seq, target_frame))
        
        print(f"数据准备完毕！共生成 {len(self.samples)} 个训练样本。")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_np, target_np = self.samples[idx]
        
        # --- 关键步骤：维度转换 ---
        # OpenCV/Matlab 是 (Height, Width, Channel)
        # PyTorch 需要 (Channel, Height, Width)
        
        # 输入: (5, 512, 512, 9) -> (5, 9, 512, 512)
        input_tensor = torch.from_numpy(input_np).permute(0, 3, 1, 2).float()
        
        # 目标: (512, 512, 3) -> (3, 512, 512)
        target_tensor = torch.from_numpy(target_np).permute(2, 0, 1).float()
        
        return input_tensor, target_tensor

# =============================================================================
# --- 3. 模型架构 (3层 ConvLSTM) ---
# =============================================================================
class PaperModel(nn.Module):
    def __init__(self):
        super(PaperModel, self).__init__()
        
        # 3层 ConvLSTM
        # 输入通道: 9, 隐藏层通道: 32, 核大小: 5x5
        self.conv_lstm = ConvLSTM(input_dim=9,
                                  hidden_dim=[32, 32, 32],
                                  kernel_size=[(5,5), (5,5), (5,5)],
                                  num_layers=3,
                                  batch_first=True,
                                  return_all_layers=False)
        
        # 输出层: 1x1 卷积, 32通道 -> 3通道 (RGB)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid() # 保证输出在 0-1 之间

    def forward(self, x):
        # ConvLSTM 输出的是列表，我们取最后一层的输出
        # output_list[0] 是最后一层的输出序列
        layer_output_list, _ = self.conv_lstm(x)
        
        # 取最后一个时间步的 hidden state
        # shape: (Batch, 32, 512, 512)
        last_state = layer_output_list[0][:, -1, :, :, :]
        
        # 通过 1x1 卷积还原为图像
        out = self.final_conv(last_state)
        out = self.sigmoid(out)
        
        return out

# =============================================================================
# --- 4. 混合损失函数 (MSE + SSIM) ---
# =============================================================================
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.05): # alpha=0.05
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # 1. MSE
        mse_loss = self.mse(preds, targets)
        
        # 2. SSIM (越高越好，所以 Loss = 1 - SSIM)
        ssim_val = ssim(preds, targets, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        
        return mse_loss + self.alpha * ssim_loss

# =============================================================================
# --- 5. 主训练循环 ---
# =============================================================================
if __name__ == "__main__":
    print(f"--- 正在使用设备: {DEVICE} ---")
    
    # 1. 加载数据
    dataset = BatteryDataset("export_images")
    if len(dataset) == 0:
        print("❌ 错误：没有找到数据，请检查 export_images 文件夹。")
        exit()
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化
    model = PaperModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = HybridLoss(alpha=0.05).to(DEVICE)
    
    print("\n--- 开始训练 (PyTorch) ---")
    
    # 3. 循环 Epoch
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.6f} ===")
        
        # 保存模型 (覆盖保存最新版)
        torch.save(model.state_dict(), "best_model_pytorch.pth")
        print("模型已保存: best_model_pytorch.pth\n")

    print("🎉 训练全部完成！")