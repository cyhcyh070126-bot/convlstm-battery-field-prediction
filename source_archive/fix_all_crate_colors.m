% =========================================================================
% === 批量修复脚本: 更新所有现有文件夹的 C-rate 颜色图 ===
% =========================================================================
clc; clear; close all;

% 1. 设置数据根目录
base_output_folder = 'export_images'; 

% 2. 获取所有子文件夹
d = dir(base_output_folder);
isub = [d(:).isdir]; 
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = []; % 去掉 . 和 ..

fprintf('发现 %d 个数据文件夹，开始检查并更新 C-rate 颜色...\n', length(nameFolds));

% 3. 定义新的颜色映射标准 (必须与 run_single_simulation.m 完全一致)
c_anchors = [0.5,  1.0,  2.0,  3.0,  4.0,  5.0]; 
rgb_anchors = [
    0.0, 0.0, 0.5;  % 0.5C: 深蓝
    0.0, 0.0, 1.0;  % 1.0C: 纯蓝
    0.0, 1.0, 1.0;  % 2.0C: 青色
    0.0, 1.0, 0.0;  % 3.0C: 纯绿 (修正点)
    1.0, 0.6, 0.0;  % 4.0C: 橙色 (修正点)
    0.6, 0.0, 0.0   % 5.0C: 深红
];

% 4. 循环处理每一个文件夹
count_fixed = 0;

for i = 1:length(nameFolds)
    folder_name = nameFolds{i};
    full_folder_path = fullfile(base_output_folder, folder_name);
    
    % --- A. 从文件夹名字中提取 C-rate ---
    % 假设文件夹名格式包含: ..._C=3_... 或 ..._C=0.5_...
    % 使用正则表达式提取 'C=' 后面的数字
    token = regexp(folder_name, 'C=([\d\.]+)', 'tokens');
    
    if isempty(token)
        fprintf('⚠️ 跳过: 无法从文件夹名 "%s" 中解析 C-rate\n', folder_name);
        continue;
    end
    
    c_rate_str = token{1}{1};
    c_rate_val = str2double(c_rate_str);
    
    % --- B. 计算正确的新颜色 (核心逻辑) ---
    if c_rate_val <= min(c_anchors)
        rgb_color = rgb_anchors(1, :);
    elseif c_rate_val >= max(c_anchors)
        rgb_color = rgb_anchors(end, :);
    else
        rgb_color = interp1(c_anchors, rgb_anchors, c_rate_val, 'linear');
    end
    rgb_color = max(0, min(1, rgb_color)); % 确保 0-1
    
    % --- C. 生成图片 ---
    image_size = [512, 512];
    R = ones(image_size) * rgb_color(1);
    G = ones(image_size) * rgb_color(2);
    B = ones(image_size) * rgb_color(3);
    solid_color_image = cat(3, R, G, B);
    
    % --- D. 覆盖保存 ---
    % 目标路径: export_images/Folder/C-rate/xC.png
    target_subdir = fullfile(full_folder_path, 'C-rate');
    
    if ~exist(target_subdir, 'dir')
        mkdir(target_subdir); % 如果不存在就创建
    end
    
    filename = fullfile(target_subdir, sprintf('%gC.png', c_rate_val));
    
    try
        imwrite(solid_color_image, filename);
        fprintf('✅ [%d/%d] 已更新: %s (C=%g -> RGB=[%.2f %.2f %.2f])\n', ...
            i, length(nameFolds), folder_name, c_rate_val, rgb_color(1), rgb_color(2), rgb_color(3));
        count_fixed = count_fixed + 1;
    catch err
        fprintf('❌ 保存失败: %s\n', filename);
    end
end

fprintf('\n--- 修复完成! 共更新了 %d 个文件夹的 C-rate 图片 ---\n', count_fixed);