
# Resolve companion modules when this historical script is invoked directly.
import sys as _sys
from pathlib import Path as _Path
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
_research_root = next(p for p in _Path(__file__).resolve().parents if p.name == "research_scripts")
_sys.path.append(str(_research_root))
_sys.path.append(str(_research_root.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# 导入你的 PyTorch 组件
from train_pytorch import PaperModel
from data_loader import load_simulation_data

# =============================================================================
# --- 配置区域 ---
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_pytorch.pth"

# 【关键修改】指定你要测试的新文件夹 (0.5C)
TARGET_FOLDER_NAME = "N=60_Lognormal_mu=2.00_sigma=0.15_R0=17.248_C=0.5_ID=69550"

def visualize_prediction():
    # 1. 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return

    print(f"正在加载 PyTorch 模型: {MODEL_PATH} ...")
    model = PaperModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 

    # 2. 构建路径
    base_data_dir = "export_images"
    test_folder = os.path.join(base_data_dir, TARGET_FOLDER_NAME)
    
    if not os.path.exists(test_folder):
        print(f"❌ 找不到文件夹: {test_folder}")
        print("请检查是否已通过 FileZilla 上传了该文件夹？")
        return

    print(f"正在测试样本: {TARGET_FOLDER_NAME}")
    
    # 3. 加载数据
    conc_seq, _ = load_simulation_data(test_folder)
    
    if conc_seq is None: 
        print("❌ 数据加载失败。")
        return

    # 4. 准备输入 (取第 10-14 帧，预测第 15 帧)
    start_idx = 10
    input_seq_np = conc_seq[start_idx : start_idx+5]
    
    # 转 PyTorch 格式
    input_tensor = torch.from_numpy(input_seq_np).float()
    input_tensor = input_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
    input_tensor = input_tensor.to(DEVICE)

    # 5. 预测
    print("AI 正在思考...")
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # 转回 Numpy
    predicted_img = prediction[0].cpu().numpy().transpose(1, 2, 0)

    # 6. 准备真值
    ground_truth = conc_seq[start_idx+5][:, :, 0:3]

    # 7. 绘图 (RGB模式)
    print("正在绘图...")
    plt.figure(figsize=(15, 5))

    # 左图: Input
    plt.subplot(1, 3, 1)
    plt.title(f"Input (t={start_idx+4})")
    input_show = input_seq_np[-1, :, :, 0:3]
    plt.imshow(input_show)
    plt.axis('off')

    # 中图: Ground Truth
    plt.subplot(1, 3, 2)
    plt.title(f"Real Future (t={start_idx+5})")
    plt.imshow(ground_truth)
    plt.axis('off')

    # 右图: AI Prediction
    plt.subplot(1, 3, 3)
    plt.title("AI Prediction (0.5C)")
    plt.imshow(predicted_img)
    plt.axis('off')

    output_file = "CHECK_new_data_result.png"
    plt.savefig(output_file)
    print(f"\n✅ 预测结果已保存为: {output_file}")
    print("请打开查看。注意观察 0.5C 下浓度变化是否比之前的 5C 缓慢？")

if __name__ == "__main__":
    visualize_prediction()