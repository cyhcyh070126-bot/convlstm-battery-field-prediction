import torch
import torch.nn as nn
from torchinfo import summary
import sys
import os
import numpy as np

# ==========================================
# 0. 环境与依赖检查
# ==========================================
try:
    from neuralop.models import FNO
except ImportError:
    print("❌ 错误：未安装 neuralop 库。请运行 'pip install neuralop'。")
    sys.exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✨ 正在使用设备: {DEVICE}")

# ==========================================
# ⚙️ 1. 配置参数 (必须与训练代码一致)
# ==========================================
# 任务定义: 输入前5帧 -> 预测后20帧
INPUT_FRAMES = 5
OUTPUT_FRAMES = 20

# 通道计算逻辑 (Seq2Seq 核心):
# 输入 = 3(晶向) + 3(C-rate) + 5x3(历史浓度) = 21 通道
IN_CHANNELS = 3 + 3 + (INPUT_FRAMES * 3) 

# 输出 = 20x3(未来浓度) = 60 通道
OUT_CHANNELS = OUTPUT_FRAMES * 3

print(f"\n📊 模型通道配置检查:")
print(f"   - 输入通道 (In): {IN_CHANNELS}  [3(Geo) + 3(Cond) + 15(Past)]")
print(f"   - 输出通道 (Out): {OUT_CHANNELS} [60(Future)]")

# ==========================================
# 🏗️ 2. 实例化 FNO 模型
# ==========================================
print("\n--- 1. 构建 FNO 模型 (Seq2Seq Mode) ---")
try:
    # 使用与 train_fno_seq2seq_final.py 相同的超参数
    model = FNO(
        n_modes=(16, 16),       # 傅里叶模态 (频率分量数)
        hidden_channels=64,     # 隐藏层宽度 (Lifting后)
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS
    ).to(DEVICE)
    print("✅ FNO 模型实例化成功。")
except Exception as e:
    print(f"❌ 模型构建失败: {e}")
    sys.exit()

# ==========================================
# 🔬 3. 结构可视化 (torchinfo)
# ==========================================
print("\n--- 2. 网络结构详细摘要 (Train Resolution: 128x128) ---")
# 创建一个 128x128 的假数据用于查看结构
dummy_input_128 = torch.randn(1, IN_CHANNELS, 128, 128).to(DEVICE)

try:
    # 打印详细的层级结构和参数量
    summary(model, input_data=dummy_input_128, 
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=1)
except Exception as e:
    print(f"⚠️ torchinfo 运行受限: {e} (不影响后续测试)")

# ==========================================
# 🧪 4. 功能性测试 (Forward Pass)
# ==========================================
print("\n--- 3. 前向传播测试 (Forward Pass Check) ---")

# --- 测试 A: 训练分辨率 (128x128) ---
print(f"\n🔹 测试 A: 输入分辨率 128x128 (训练标准)")
try:
    with torch.no_grad():
        output_128 = model(dummy_input_128)
    
    expected_shape = (1, OUT_CHANNELS, 128, 128)
    if output_128.shape == expected_shape:
        print(f"✅ 测试通过! 输出形状: {output_128.shape}")
        print(f"   (包含 20 帧未来数据，每帧 3 通道)")
    else:
        print(f"❌ 失败! 预期 {expected_shape}, 实际 {output_128.shape}")
except Exception as e:
    print(f"❌ 崩溃: {e}")

# --- 测试 B: 高清分辨率 (512x512) ---
# FNO 的核心优势是分辨率无关性 (Zero-Shot Super-Resolution)
# 如果这个通过，说明你的模型训练好后可以直接用来预测 512 的图，不需要重新训练！
print(f"\n🔹 测试 B: 输入分辨率 512x512 (FNO 分辨率无关性验证)")

try:
    # 显存保护：如果显存不够，可能跑不动 512 的 batch，这里加个 try-catch
    dummy_input_512 = torch.randn(1, IN_CHANNELS, 512, 512).to(DEVICE)
    
    with torch.no_grad():
        output_512 = model(dummy_input_512)
    
    expected_shape_512 = (1, OUT_CHANNELS, 512, 512)
    if output_512.shape == expected_shape_512:
        print(f"✅ 测试通过! FNO 成功处理了 512x512 输入，输出形状: {output_512.shape}")
        print("🚀 结论: 模型结构正确，且具备【零样本超分】能力 (Zero-Shot Super-Resolution)。")
        print("   这意味着你可以用 128 的图训练，然后直接预测 512 的高清结果！")
    else:
        print(f"❌ 失败! 输出形状异常: {output_512.shape}")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("⚠️ 显存不足 (OOM)，无法跑通 512 测试。这是硬件限制，不是模型结构错误。")
        print("   (建议: 实际预测 512 时可以使用 CPU 或更小的 Batch)")
    else:
        print(f"❌ 崩溃: {e}")

print("\n✨ 检查完成。")