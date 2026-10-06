 % =========================================================================
% ===       自动化装配线 (V5): run_single_simulation.m             ===
% =========================================================================
%
%
% =========================================================================

clc;
clear all; % 只有这个主脚本可以 'clear all'
close all;
tic; % 开始计时
disp('--- 自动化装配线 (V5) 已启动 ---');


try
    % =====================================================================
    % --- 步骤 0: 切换到你的工作目录 ---
    % =====================================================================
    WORK_DIR = fileparts(mfilename('fullpath'));
    cd(WORK_DIR);
    fprintf('已切换到工作目录: %s\n', WORK_DIR);

    % --- 准备 COMSOL Livelink (再次导入) ---
    import com.comsol.model.*
    import com.comsol.model.util.*



fprintf('-----------------------------------------------------------\n');
    % =====================================================================
    % --- 模块 1: 运行 "微观结构工厂" (包含 clear 命令) ---
    % =====================================================================
    disp('步骤 1: 运行 main_D21origin.m ---');
    % (此脚本会运行它自己的 'clear', 'clc', 'close all')
    % (这会擦除工作区，但没关系，因为我们还没定义 C-rate)
    main_D21origin;
    disp(' 微观结构 (geometry_for_comsol.mat) 生成完毕。');
    

% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------

      % --- 【【【【 V18 新增：捕获 "工厂" 脚本的输出 】】】】 ---
    try
        N_val = N; % 捕获晶粒数 (e.g., 80)
        R0_val = R0; % 捕获外径 (e.g., 5.64)
        command_val = command; % <-- 【【 采纳你的建议：捕获 command 】】
        
        % --- 【【 V18：将 command 转换为字符串 】】 ---
        switch command_val
            case 1
                dist_name_simple = 'Weibull';
            case 2
                dist_name_simple = 'Lognormal';
            case 3
                dist_name_simple = 'Normal';
            otherwise
                dist_name_simple = 'UnknownDist';
        end
        % --- V18 结束 ---
        
        fprintf('... 已捕获参数: N=%d, R0=%.2f, Dist=%s (Command=%d)\n', N_val, R0_val, dist_name_simple, command_val);
        fprintf('-----------------------------------------------------------\n');
        fprintf('-----------------------------------------------------------\n');
    catch capture_err
        disp('警告: 无法从 "main_D21origin.m" 捕获参数 (N, R0, command)。');
        disp(capture_err.message);
        % B方案: 使用一个默认的 "错误" 文件夹名称
        N_val = 0; R0_val = 0; dist_name_simple = 'CAPTURA_FALHOU';
    end
    % --- 【V18 新增结束】 ---

   



% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------




    % =====================================================================
    % --- 模块 2: 运行 "COMSOL 物理引擎" ---
    % =====================================================================
    disp(' 步骤 2: 运行 build_full_comsol_model.m ---');
    % (此脚本会 'clear model' 并创建新的 'model' 变量)
    build_full_comsol_model; 
 
    disp('... COMSOL 模型构建脚本执行完毕。');

    % =====================================================================
    % --- 【【【【 错误验证 】】】】 ---
    % =====================================================================
    if ~exist('model', 'var')
        error('CRITICAL_ERROR: "build_full_comsol_model.m" 未能成功创建 "model" 变量。请检查该脚本的错误日志。');
    end
    disp('... 成功验证 "model" 变量存在。');
    % =====================================================================
    % =====================================================================
    % --- 模块 3: 【【【 你的逻辑：现在才定义变量 】】】 ---
    % =====================================================================
    
    % (现在 'clear' 已经运行完毕，我们在这里定义变量是 100% 安全的)
    %

% ---------------------------------------------------------------------------------------









    % 【根据你的要求】:
    TLIST_TO_RUN = 'range(0,100,2400)'; % (你要求的时间步长)
    
    % 【【【【 在这里修改 C-rate 】】】】    
    % 【【【【 在这里修改 C-rate 】】】】








    C_RATE_TO_RUN =5; 
              









     % 【【【【 在这里修改 C-rate 】】】】
    % 【【【【 在这里修改 C-rate 】】】】
    % 【【【【 C-rate 修改结束 】】】】
    


















% ---------------------------------------------------------------------------------------


% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------

    fprintf('设置参数: C-rate = %d, T_list = %s\n', C_RATE_TO_RUN, TLIST_TO_RUN);
    % --- 开始后台计算 (model.study.run) ---
    fprintf('-----------------------------------------------------------\n');
    fprintf('-----------------------------------------------------------\n');
    fprintf('-----------------------------------------------------------\n');
    disp('步骤 3: 覆盖参数并开始后台计算 (model.study.run) ---');




% ---------------------------------------------------------------------------------------

    % --- a. 覆盖 C-rate ---
    model.param('par2').set('c_rate', string(C_RATE_TO_RUN));
    fprintf('... 参数已覆盖: C-rate 设置为 %d\n', C_RATE_TO_RUN);



% ---------------------------------------------------------------------------------------

    % --- b. 覆盖时间步长 ---
    model.study('std1').feature('time').set('tlist', TLIST_TO_RUN);
    fprintf('... 参数已覆盖: tlist 设置为 %s\n', TLIST_TO_RUN);




% ---------------------------------------------------------------------------------------
    fprintf('-----------------------------------------------------------\n');
        fprintf('-----------------------------------------------------------\n');
% --- c. (V15) 运行标准的研究 (这才是正确的后台命令) ---
    disp('...现在开始仿真计算。');

    model.study('std1').run();
    
    disp('... 仿真计算完成。');
  
    fprintf('-----------------------------------------------------------\n');  
        fprintf('-----------------------------------------------------------\n');
% ---------------------------------------------------------------------------------------



 % =====================================================================
    % --- 模块 4: 自动化数据导出 (V21 - 动态文件夹 & 泰森多边形图) ---
    % =====================================================================
    disp('--- 步骤 4: 自动导出图像 (V21 最终版) ---');

    % --- 【【【【 V21 修正：在这里定义临时文件夹路径 】】】】 ---
    % (在 'clear all' 之后定义，100% 安全)
    % ！！ 确保这个路径与你在 main_D21origin.m 中设置的 temp_fig_folder 一致 ！！

    WORK_DIR = fileparts(mfilename('fullpath'));
    MAIN_D21ORIGIN_TEMP_FIG_FOLDER = fullfile(WORK_DIR, 'temp_figs');
    % --- V21 修正结束 ---

% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------


    % --- a. 【【 V18 修正：创建动态文件夹 】】】 ---
 base_output_folder = 'export_images';




random_id = randi(99999); % 生成一个 1 到 99999 之间的随机整数

run_folder_name = sprintf('N=%d_%s_mu=%.2f_sigma=%.2f_R0=%.3f_C=%g_ID=%05d', N_val, dist_name_simple, mu, sigma, R0_val, C_RATE_TO_RUN, random_id);



% --- 【【 V19 新增：泰森多边形图的子文件夹 】】】 ---
% --- 【【 已按 1, 2, 3 顺序重命名 】】】 ---
folder_voronoi_figs = fullfile(base_output_folder, run_folder_name, '3_Voronoi_Geometry');
folder_c = fullfile(base_output_folder, run_folder_name, '1_Concentration');
folder_stress = fullfile(base_output_folder, run_folder_name, '2_Stress');
folder_c_rate = fullfile(base_output_folder, run_folder_name, 'C-rate'); % <-- 【【【【 新增 (来自 image_3d0100.png) 】】】】



    if ~exist(folder_voronoi_figs, 'dir')
       mkdir(folder_voronoi_figs);
    end
    if ~exist(folder_c, 'dir')
       mkdir(folder_c);
    end
    if ~exist(folder_stress, 'dir')
       mkdir(folder_stress);
    end
    if ~exist(folder_c_rate, 'dir') % <-- 【【【【 新增 】】】】
   mkdir(folder_c_rate);      % <-- 【【【【 新增 】】】】
    end
    fprintf('... 图像将分类保存到: %s\n', fullfile(base_output_folder, run_folder_name));
    % --- V18/V19 修正结束 ---



try
    disp('... C-rate 纯色图 ...');

    % 1. 获取 C-rate 值
    c_rate_val = C_RATE_TO_RUN; %
    %    根据 C-rate 选择一个 RGB 颜色 (0-1范围)




% -----------------------------------------------------------
    % 2. 【【【【 连续颜色映射 (V22 升级版) 】】】】
    %    使用插值法，覆盖所有 C-rate，让 3C, 4C 等都有不同颜色
    % -----------------------------------------------------------
    
    % A. 定义关键锚点 (C-rate 数值)
    % 我们定义一条从低倍率到高倍率的路线
    c_anchors = [0.5,  1.0,  2.0,  3.0,  4.0,  5.0]; 
    
    % B. 定义对应的颜色锚点 (RGB)
    % 每一行对应上面的一个 C-rate
    rgb_anchors = [
        0.0, 0.0, 0.5;  % 0.5C: 深海军蓝 (Dark Navy)
        0.0, 0.0, 1.0;  % 1.0C: 纯蓝 (Blue)
        0.0, 1.0, 1.0;  % 2.0C: 青色 (Cyan)
        0.0, 1.0, 0.0;  % 3.0C: 纯绿 (Green) -> 【解决了3C的问题】
        1.0, 0.6, 0.0;  % 4.0C: 橙色 (Orange) -> 【解决了4C的问题】
        0.6, 0.0, 0.0   % 5.0C: 深红 (Dark Red)
    ];

    % C. 计算当前 C-rate 的颜色
    if c_rate_val <= min(c_anchors)
        % 如果小于等于 0.5，使用深蓝
        rgb_color = rgb_anchors(1, :);
    elseif c_rate_val >= max(c_anchors)
        % 如果大于等于 5，使用深红
        rgb_color = rgb_anchors(end, :);
    else
        % 【关键】如果在中间 (比如 3.5)，使用线性插值自动混合颜色
        rgb_color = interp1(c_anchors, rgb_anchors, c_rate_val, 'linear');
    end
    


    % 确保颜色值在 0-1 之间 (防止插值溢出)
    rgb_color = max(0, min(1, rgb_color));
    fprintf('...... C-rate = %g, 计算出的 RGB = [%.2f %.2f %.2f]\n', c_rate_val, rgb_color);




    % 3. 创建 512x512 图像
    image_size = [512, 512];
    R_channel = ones(image_size) * rgb_color(1);
    G_channel = ones(image_size) * rgb_color(2);
    B_channel = ones(image_size) * rgb_color(3);
    solid_color_image = cat(3, R_channel, G_channel, B_channel);





    % 4. 保存图像 (文件名格式如 0.5C.png, 5C.png)
    filename_c_rate = fullfile(folder_c_rate, sprintf('%gC.png', c_rate_val));
    imwrite(solid_color_image, filename_c_rate);

    fprintf('...... 已保存: %s\n', filename_c_rate);



 fprintf('-----------------------------------------------------------\n');
  



catch c_rate_err
    disp('--- 警告: 生成 C-rate 纯色图失败 ---');
    disp(c_rate_err.message);
end
% --- 【【【【 新增模块 V3 结束 】】】】 ---




 fprintf('------------------------------------------------\n');      
    % --- 【【【 V19 新增：移动 main_D21origin.m 生成的 PNGs 】】】 ---
    disp('... 正在移动 "泰森多边形" 相关的PNG 图像 ...');
    voronoi_temp_files = dir(fullfile(MAIN_D21ORIGIN_TEMP_FIG_FOLDER, '*.png'));


    if ~isempty(voronoi_temp_files)
        for k = 1:length(voronoi_temp_files)
            source_file = fullfile(MAIN_D21ORIGIN_TEMP_FIG_FOLDER, voronoi_temp_files(k).name);
            destination_file = fullfile(folder_voronoi_figs, voronoi_temp_files(k).name);
            movefile(source_file, destination_file, 'f'); % 移动文件 (f = 强制)
            fprintf('...... 已移动: %s\n', voronoi_temp_files(k).name);
        end
        % 可选：不删除临时文件夹，以便调试
        % rmdir(MAIN_D21ORIGIN_TEMP_FIG_FOLDER); 
        % disp('... 临时泰森多边形文件夹已清理。');
    else
        disp('... 未找到 "泰森多边形" 相关的临时 PNG 图像。');
    end
    % --- V19 新增结束 ---



    
% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------
% --- b. 定义绘图组 (Plot Groups) ---


    % 绘图组 1: 浓度 (c)
    pg_c = model.result.create('pg_c', 'PlotGroup2D');
    pg_c.label('浓度 (c)');
    surf_c = pg_c.feature.create('surf_c', 'Surface');
    surf_c.set('expr', 'c'); 
    surf_c.set('colortable', 'Rainbow'); % 彩色图

%surf_c.set('colortable', 'GrayScale');% 灰度图




    % --- 【【【【 V13 修正 (来自 COMSOL_Log_V12.m) 】】】】 ---
    surf_c.set('rangecoloractive', true); % <-- 你找到的! (☑ 手动控制)
    surf_c.set('rangecolormax', '4.5E4');  % <-- 你找到的! (最大值)
    % surf_c.set('rangecolormin', '0');    % (可选, 0 是默认值)
    % --- 修正结束 ---





    % 绘图组 2: von Mises 应力 (solid.mises)
    pg_stress = model.result.create('pg_stress', 'PlotGroup2D');
    pg_stress.label('von Mises 应力');
    surf_stress = pg_stress.feature.create('surf_stress', 'Surface');
    surf_stress.set('expr', 'solid.mises'); 



    surf_stress.set('colortable', 'Prism'); % 彩色图

 
   %surf_stress.set('colortable', 'GrayScale'); % 灰度图




    % --- 【【【【 V13 修正 (来自 COMSOL_Log_V12.m) 】】】】 ---
    surf_stress.set('rangecoloractive', true); % <-- 你找到的! (☑ 手动控制)
    surf_stress.set('rangecolormax', '5E8');    % <-- 你找到的! (最大值)
    % surf_stress.set('rangecolormin', '0');  % (可选, 0 是默认值)
    % --- 修正结束 ---




% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------
    % --- c. 定义图像导出器 ---
    img_export = model.result.export.create('img1', 'Image');
    
   
    img_export.set('size', 'manualweb'); % <-- 1. (设置预设为 "手动")
    img_export.set('lockratio', 'off');    % <-- 2. (解锁宽高比)
    img_export.set('width', 512);          % <-- 3. (设置宽度, 数字)
    img_export.set('height', 512);         % <-- 4. (设置高度, 数字)
 



% 【【【【 新增：设置分辨率为 96 DPI 】】】】
    img_export.set('resolution', 120);
    % 【【【【 新增结束 】】】】

    img_export.set('unit', 'px'); % (这一行现在是多余的, 但保留也无妨)
   


    % --- d. 循环遍历所有时间步并导出 ---
            fprintf('------------------------------------------------\n');      
                    fprintf('------------------------------------------------\n');      
    disp('... 开始循环导出时间帧 ...');
    % 我们不再调用 mphgetsolinfo，因为 'clear all' 破坏了路径。
    % 我们直接使用我们已知的、在 TLIST_TO_RUN 中定义的向量。
    tlist_values = 0:100:2400;
    fprintf('... 将使用预定义的时间列表: %s (共 %d 帧)\n', TLIST_TO_RUN, length(tlist_values));
    % --- 修正结束 ---

    for i = 1:length(tlist_values)
        current_time = tlist_values(i);
        sol_index = i; % COMSOL 索引从 1 开始
        
        % --- 导出浓度 ---
        pg_c.set('solnum', sol_index); 
        img_export.set('plotgroup', 'pg_c'); 
        % --- 【【 V18 修正：使用新路径 】】 ---
        filename_c = fullfile(folder_c, sprintf('concentration_t%05.0f.png', current_time));
        img_export.set('filename', filename_c);
        img_export.run(); 

        % --- 导出应力 ---
        pg_stress.set('solnum', sol_index); 
        img_export.set('plotgroup', 'pg_stress'); 
        filename_stress = fullfile(folder_stress, sprintf('stress_mises_t%05.0f.png', current_time));
        img_export.set('filename', filename_stress);
        img_export.run(); 
        
        fprintf('... 已导出: t = %.0f s (浓度 & 应力)\n', current_time);
    end
    
    disp('--- 步骤 4 自动化导出完成! ---');
    fprintf('-----------------------------------------------------------\n');


% --- 步骤 5: (可选) 正在保存包含【完整解】的 .mph 模型文件 ---
    disp('--- 步骤 5: 正在保存包含【完整解】的 .mph 模型文件 ---');

    % 使用你在步骤 4a (第 184 行) 中创建的文件夹名称和路径
    mph_filename = fullfile(base_output_folder, run_folder_name, [run_folder_name '.mph']);

    mphsave(model, mph_filename);

    %fprintf('... .mph 模型 (包含解) 已保存到: %s\n', mph_filename);
    fprintf('-----------------------------------------------------------\n');
    fprintf('-----------------------------------------------------------\n');
    fprintf('-----------------------------------------------------------\n');


fprintf('... 本次运行的 ID 为: %05d\n', random_id); % <-- 【【【【 新增的日志行 】】】】
    

    fprintf('-----------------------------------------------------------\n');
       fprintf('-----------------------------------------------------------\n');
        fprintf('-----------------------------------------------------------\n');



        % --- 【【【【 新增：打印 mu 和 sigma 】】】】 ---
fprintf('... 本次运行均值 (mu) 为: %g\n', mu);
fprintf('... 本次运行标准差 (sigma) 为: %g\n', sigma);
% --- 【【【【 修改结束 】】】】 ---



fprintf('-----------------------------------------------------------\n');
catch e
    disp('--- 发生严重错误 ---');
    disp(e.message);
    if ~isempty(e.stack)
        fprintf('错误发生在文件 %s 的第 %d 行\n', e.stack(1).file, e.stack(1).line);
    end
end


    fprintf('-----------------------------------------------------------\n');

toc; % 结束计时
disp('--- 自动化脚本运行结束 ---');
 