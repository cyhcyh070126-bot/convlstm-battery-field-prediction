import numpy as np
import cv2
import os
import glob

def load_simulation_data(sample_folder_path, mode='concentration'):
    """
    加载一个样本文件夹的数据。
    Args:
        sample_folder_path: 文件夹路径
        mode: 'concentration' (只加载浓度), 'stress' (只加载应力), 或 'both'
    Returns:
        conc_seq, stress_seq (如果不加载某项，则返回 None)
        数据形状: (25, 512, 512, 9)
    """
    # print(f"--- 正在处理: {os.path.basename(sample_folder_path)} ---") # 减少刷屏，可注释
    
    TARGET_SIZE = (512, 512)

    # === 1. 加载 静态通道 B (晶粒取向图) ===
    try:
        orientation_file = glob.glob(os.path.join(sample_folder_path, "3_Voronoi_Geometry", "05_*.png"))[0]
        orientation_map = cv2.resize(cv2.cvtColor(cv2.imread(orientation_file), cv2.COLOR_BGR2RGB), TARGET_SIZE)
        orientation_map = orientation_map.astype(np.float32) / 255.0
    except IndexError:
        print(f"!! 错误: 未找到晶粒取向图: {sample_folder_path}")
        return None, None

    # === 2. 加载 静态通道 C (C-rate 图) ===
    try:
        c_rate_file = glob.glob(os.path.join(sample_folder_path, "C-rate", "*.png"))[0]
        c_rate_map = cv2.resize(cv2.cvtColor(cv2.imread(c_rate_file), cv2.COLOR_BGR2RGB), TARGET_SIZE)
        c_rate_map = c_rate_map.astype(np.float32) / 255.0
    except IndexError:
        print(f"!! 错误: 未找到 C-rate 图片: {sample_folder_path}")
        return None, None

    # === 3. 加载 动态通道 (浓度/应力) ===
    concentration_sequence = []
    stress_sequence = []

    # 获取文件列表
    concentration_path = os.path.join(sample_folder_path, "1_Concentration") 
    stress_path = os.path.join(sample_folder_path, "2_Stress") 
    
    # 只需要数量对齐即可
    conc_files = sorted(glob.glob(os.path.join(concentration_path, "*.png")))
    stress_files = sorted(glob.glob(os.path.join(stress_path, "*.png")))
    
    if not conc_files:
        print(f"!! 错误: 未找到数据文件")
        return None, None
        
    num_frames = len(conc_files)

    for i in range(num_frames):
        # --- 按需加载浓度 ---
        if mode == 'concentration' or mode == 'both':
            img = cv2.resize(cv2.cvtColor(cv2.imread(conc_files[i]), cv2.COLOR_BGR2RGB), TARGET_SIZE)
            img = img.astype(np.float32) / 255.0
            # 堆叠: [浓度(3), 晶向(3), C-rate(3)] -> (512, 512, 9)
            stack = np.concatenate([img, orientation_map, c_rate_map], axis=-1)
            concentration_sequence.append(stack)

        # --- 按需加载应力 ---
        if mode == 'stress' or mode == 'both':
            img = cv2.resize(cv2.cvtColor(cv2.imread(stress_files[i]), cv2.COLOR_BGR2RGB), TARGET_SIZE)
            img = img.astype(np.float32) / 255.0
            # 堆叠: [应力(3), 晶向(3), C-rate(3)] -> (512, 512, 9)
            stack = np.concatenate([img, orientation_map, c_rate_map], axis=-1)
            stress_sequence.append(stack)

    final_conc = np.array(concentration_sequence) if concentration_sequence else None
    final_stress = np.array(stress_sequence) if stress_sequence else None
    
    return final_conc, final_stress