import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2

# 导入你的 PyTorch 组件
from train_pytorch import PaperModel
from data_loader import load_simulation_data

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_pytorch.pth"

def visualize_prediction():
    # 1. 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        print("请先运行 'python train_pytorch.py' 进行训练！")
        return

    print(f"正在加载 PyTorch 模型: {MODEL_PATH} ...")
    
    # 2. 初始化模型并加载权重
    model = PaperModel().to(DEVICE)
    # map_location 确保即使在没有 GPU 的电脑上也能加载(虽然你有)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # 切换到评估模式

    # 3. 随机加载一个样本文件夹
    base_data_dir = "export_images"
    all_folders = [os.path.join(base_data_dir, f) for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    
    if not all_folders:
        print("❌ 没有找到数据文件夹！")
        return

    test_folder = random.choice(all_folders)
    print(f"正在测试样本: {os.path.basename(test_folder)}")
    
    # 加载数据 (HWC 格式, RGB, 9通道)
    # conc_seq 形状: (25, 512, 512, 9)
    conc_seq, _ = load_simulation_data(test_folder)
    
    if conc_seq is None: return

    # 4. 【关键修改】随机截取一段视频 (5帧输入 + 1帧目标)
    total_frames = conc_seq.shape[0] # 应该为 25
    # 我们需要连续 6 帧 (5 输入 + 1 目标)，所以起始点最大只能到 19 (25-6)
    max_start_idx = total_frames - 5 - 1
    
    # 随机选择起始点
    start_idx = random.randint(0, max_start_idx)
    print(f"本次随机截取时间段: Input t={start_idx}~{start_idx+4} --> Target t={start_idx+5}")

    # 取出 numpy 数据: (5, 512, 512, 9)
    input_seq_np = conc_seq[start_idx : start_idx+5]
    
    # --- 转换为 PyTorch 格式 (B, T, C, H, W) ---
    # (1, 5, 512, 512, 9) -> (1, 5, 9, 512, 512)
    input_tensor = torch.from_numpy(input_seq_np).float()
    input_tensor = input_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
    input_tensor = input_tensor.to(DEVICE)

    # 5. 模型预测
    print("AI 正在思考...")
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # prediction 形状: (1, 3, 512, 512)
    # 转回 Numpy: (512, 512, 3)
    predicted_img = prediction[0].cpu().numpy().transpose(1, 2, 0)

    # 6. 准备真实值
    ground_truth = conc_seq[start_idx+5][:, :, 0:3]

    # 7. 绘图 (直接画 RGB，不需要再转换了！)
    print("正在绘图...")
    plt.figure(figsize=(15, 5))

    # --- 左图: Input (Last Frame) ---
    plt.subplot(1, 3, 1)
    plt.title(f"Input (t={start_idx+4})")
    # 直接取前3通道 (已经是 RGB 了)
    input_show = input_seq_np[-1, :, :, 0:3]
    plt.imshow(input_show)
    plt.axis('off')

    # --- 中图: Ground Truth ---
    plt.subplot(1, 3, 2)
    plt.title(f"Real Future (t={start_idx+5})")
    plt.imshow(ground_truth) # 已经是 RGB
    plt.axis('off')

    # --- 右图: AI Prediction ---
    plt.subplot(1, 3, 3)
    plt.title("AI Prediction")
    plt.imshow(predicted_img) # 已经是 RGB
    plt.axis('off')

    output_file = "CHECK_pytorch_result.png"
    plt.savefig(output_file)
    print(f"\n✅ 预测结果已保存为: {output_file}")
    print("请打开查看。如果是红芯蓝边，且形状准确，说明模型成功了！")

if __name__ == "__main__":
    visualize_prediction()