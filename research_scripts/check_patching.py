
# Resolve companion modules when this historical script is invoked directly.
import sys as _sys
from pathlib import Path as _Path
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
_research_root = next(p for p in _Path(__file__).resolve().parents if p.name == "research_scripts")
_sys.path.append(str(_research_root))
_sys.path.append(str(_research_root.parent))
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# === 修正点：去掉 .py 后缀 ===
from 训练代码.单帧预测模型.train_pytorch import RAMBatteryDataset as PatchedBatteryDataset 

def check_dataset_health():
    print("--- 🏥 开始切片数据体检 ---")
    
    # 1. 配置路径
    base_data_dir = "export_images" 
    if not os.path.exists(base_data_dir):
        print(f"❌ 错误：找不到文件夹 {base_data_dir}")
        return

    # 获取所有子文件夹
    all_folders = [os.path.join(base_data_dir, f) for f in os.listdir(base_data_dir) 
                   if os.path.isdir(os.path.join(base_data_dir, f))]
    
    if len(all_folders) == 0:
        print("❌ 错误：export_images 文件夹是空的！")
        return

    # 为了快速测试，我们只用 1 个文件夹来测
    test_folders = all_folders[:1] 
    print(f"📂 正在测试加载 {len(test_folders)} 个文件夹的数据...")

    # 2. 实例化 Dataset
    dataset = PatchedBatteryDataset(test_folders, is_training=False)
    
    # 3. 检查样本总数
    # 理论值 = 文件夹数 * 20 (时间序列) * 16 (空间切块)
    expected_len = sum(max(0, len(item["conc"]) - 5) * 16 for item in dataset.data_cache)
    print(f"📊 Dataset 长度: {len(dataset)}")
    
    if len(dataset) == expected_len:
        print(f"✅ 数量检查通过！(预期 {expected_len}, 实际 {len(dataset)})")
    else:
        print(f"⚠️ 数量可能有误，请检查滑动窗口或切块逻辑。")

    # 4. 检查单个样本的形状
    if len(dataset) > 0:
        sample_input, sample_target = dataset[0]
        
        print("\n📦 单个样本形状检查:")
        print(f"   Input Shape : {sample_input.shape}")
        print(f"   Target Shape: {sample_target.shape}")
        
        # 验证 Input 形状: (5, 9, 128, 128)
        if sample_input.shape == (5, 9, 128, 128):
            print("✅ Input 维度正确 (Time=5, Channel=9, 128x128)")
        else:
            print(f"❌ Input 维度错误！期望 (5, 9, 128, 128)")

        # 验证 Target 形状: (3, 128, 128)
        if sample_target.shape == (3, 128, 128):
            print("✅ Target 维度正确 (Channel=3, 128x128)")
        else:
            print(f"❌ Target 维度错误！期望 (3, 128, 128)")

        # 5. 检查数值范围
        print("\n🧮 数值范围检查:")
        print(f"   Input Max: {sample_input.max():.4f}, Min: {sample_input.min():.4f}")
        if sample_input.max() <= 1.0 and sample_input.min() >= 0.0:
            print("✅ 数据已归一化 (0~1 之间)")
        else:
            print("⚠️ 警告：数据似乎未归一化")

        # 6. 可视化检查
        try:
            first_frame_conc = sample_input[0, 0, :, :].numpy() # 取第0帧，第0通道
            plt.figure(figsize=(4, 4))
            plt.imshow(first_frame_conc, cmap='jet')
            plt.title("Patch Preview (128x128)")
            plt.colorbar()
            plt.show()
            print("\n🖼️ 已显示第一个 Patch 的预览图，请确认它看起来像一个局部切片。")
        except Exception as e:
            print(f"\n⚠️ 无法绘图: {e}")
    else:
        print("❌ Dataset 是空的，无法检查样本形状。")

if __name__ == "__main__":
    check_dataset_health()