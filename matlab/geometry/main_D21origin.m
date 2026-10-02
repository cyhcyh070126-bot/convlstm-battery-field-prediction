% Original geometry-generation algorithm; called by run_single_simulation.
tic % 开始计时
% MATLAB uses its configured BLAS library.

temp_fig_folder = fullfile(pwd, 'temp_figs');
if ~exist(temp_fig_folder, 'dir')
    mkdir(temp_fig_folder);
end

%parpool('local',core_number);%core_number表示调用的核数
N=60;% 颗粒数目
ndm = 2;

% sigma_r=0.2;%标准差
phi_0=0.95;% Initial volume fraction
beta =521; % 控制粒子移动权重
Tk=200000;

% ftol=5.0e-3; ttol=2.0e-4; deltat0=.4;
ftol=7.8e-4; ttol=1.48e-4; deltat0=.1;
if isempty(workflow_random_seed)
    rng('shuffle');
else
    rng(workflow_random_seed, 'twister');
end

% ---------------------------- Generate Radius Distribution ------------------------------------------------- %
% ------- 1 更改均值和标准差
% ------- 2. 选择分布
% ------- 3. 记录R0

mu=2; % 均值
sigma=0.35; % 标准差

% ------------- ------ ------ ------ ------
command=3; % command= 1 对应Weibull distribution；
           % command= 2 对应Lognormal distribution；
           % command= 3 对应normal distribution；

%(:)强迫表示为列向量
% ;表示垂直拼接
%,表示水平拼接

% ---------------------------- Weibull distribution ----------------------------- %
if command==1
    distribution_name='distribution_weibull';
    sigma2 = sigma^2;
    lambda_func = @(k) mu / gamma(1 + 1/k);
    variance_func = @(k, lambda) lambda^2 * (gamma(1 + 2/k) - gamma(1 + 1/k)^2);
    error_func = @(k) (variance_func(k, lambda_func(k)) - sigma2)^2;
    k_opt = fminsearch(error_func, 1.5);
    lambda = mu / gamma(1 + 1/k_opt);

% ------------------------------分箱法 (Binning Method) ----------------------------- %
    sampling_method_name = '(Binning Method)'; % 定义方法名称
    Rs_ini = wblrnd(lambda, k_opt, 1000*N, 1);
    x_bins = linspace(0, max(Rs_ini), 20);
    Rs=[];
    for i=1:length(x_bins)-1
        Rs_qu=Rs_ini(Rs_ini >= x_bins(i) & Rs_ini < x_bins(i+1));
        if isempty(Rs_qu), continue; end
        bin_center = (x_bins(i+1) + x_bins(i)) / 2;
        pdf_val = (k_opt / lambda) * (bin_center / lambda).^(k_opt - 1) .* exp(-(bin_center / lambda).^k_opt);
        num=round(N*(x_bins(i+1)-x_bins(i))*pdf_val);
        if num == 0, continue; end
        random_element = randsample(Rs_qu, num, true);
        Rs=[Rs; random_element];
    end

% ---------------------------- 直接随机采样 --------------------------- %
    % --- 直接随机采样 (Direct Random Sampling) ---

    % sampling_method_name = '(Direct Sampling)'; % 如果使用此方法，请取消本行注释
    % Rs = wblrnd(lambda, k_opt, N, 1);
    % --- 样本数量调整 ---
    if length(Rs)>N
        deleteIdx = randsample(1:length(Rs), length(Rs)-N);
        Rs(deleteIdx) = [];
    end
    if length(Rs)<N
        Supplement_element=randsample(Rs_ini, N-length(Rs), true);
        Rs=[Rs; Supplement_element];
    end

% 绘制直方图
    figure('Visible', 'off');
    hold on; box on;
    histogram(Rs, 10, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 0.8);
    x = linspace(0.5, max(Rs)+0.5, 20);
    pdf_theory = (k_opt / lambda) * (x / lambda).^(k_opt - 1) .* exp(-(x / lambda).^k_opt);
    plot(x, pdf_theory, 'r', 'LineWidth', 2);

    % 动态设置标题
    title(['Weibull Distribution ', sampling_method_name]);

    xlabel('Particle Radius');
    ylabel('Probability Density');
    set(gca,'LineWidth',1.5);  set(gca,'FontSize',16);
    legend('Sampled Data', 'Theoretical PDF', 'location', 'northeast', 'LineWidth',0.8);
    hold off;
    Rs = Rs(:);
    saveas(gcf, fullfile(temp_fig_folder, '01_Radius_Distribution.png'));
    drawnow;
    close(gcf); % <-- 新增：保存后关闭，释放内存

% ---------------------------- Lognormal distribution ----------------------------- %
elseif command==2

distribution_name='distribution_lognormal';
    mu_log = log((mu^2)/sqrt(sigma^2+mu^2));
    sigma_log = sqrt(log(sigma^2/(mu^2)+1));

 % ------------------------------分箱法 (Binning Method) ----------------------------- %
    % --- 分箱法 (Binning Method) ---
      sampling_method_name = '(Binning Method)'; % 如果使用此方法，请取消本行注释
      Rs_ini=lognrnd(mu_log,sigma_log,1000*N,1); % 为样本调整提供数据源
      x_bins = linspace(min(Rs_ini), max(Rs_ini), 25);
      Rs=[];
      for i=1:length(x_bins)-1
          Rs_qu=Rs_ini(Rs_ini >= x_bins(i) & Rs_ini < x_bins(i+1));
          if isempty(Rs_qu), continue; end
          bin_center = (x_bins(i+1) + x_bins(i)) / 2;
          pdf_val = (1./(bin_center*sigma_log*sqrt(2*pi))) .* exp(-(log(bin_center) - mu_log).^2 / (2*sigma_log^2));
          num=round(N*(x_bins(i+1)-x_bins(i))*pdf_val);
          if num > 0
             random_element = randsample(Rs_qu, num, true);
             Rs=[Rs; random_element];
          end
     end

% ---------------------------- 直接随机采样 --------------------------- %
   % ---------------------------- 直接随机采样 --------------------------- %
    % --- 直接随机采样 ---不用就注释以下两行
    % sampling_method_name = '(Direct Sampling)'; % 定义方法名称
    % Rs = lognrnd(mu_log, sigma_log, N, 1);

    % --- 样本数量调整 ---
    if length(Rs)>N
        deleteIdx = randsample(1:length(Rs), length(Rs)-N);
        Rs(deleteIdx) = [];
    end
    if length(Rs)<N
        Supplement_element=randsample(Rs_ini, N-length(Rs), true);
        Rs=[Rs; Supplement_element];
    end

% --- 绘制对数正态分布的直方图 ---
    figure('Visible', 'off');
    histogram(Rs, 10, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 0.8);
    hold on; box on;
    x_theory = linspace(min(Rs)*0.8, max(Rs)*1.2, 100);
    pdf_theory = (1./(x_theory*sigma_log*sqrt(2*pi))) .* exp(-(log(x_theory) - mu_log).^2 / (2*sigma_log^2));
    plot(x_theory, pdf_theory, 'r-', 'LineWidth', 2);

    % *** 核心修改：动态设置标题 ***
    title(['Log-Normal Distribution ', sampling_method_name]);

    xlabel('Particle Radius (\mum)');
    ylabel('Probability Density');
    set(gca,'LineWidth',1.5);  set(gca,'FontSize',16);
    legend('Sampled Data', 'Theoretical PDF','location', 'northeast');
    hold off;
    Rs = Rs(:);
    saveas(gcf, fullfile(temp_fig_folder, '01_Radius_Distribution.png'));
    drawnow;
    close(gcf); % <-- 新增：保存后关闭，释放内存

% ---------------------------- normal distribution ----------------------------- %
elseif command==3
    distribution_name='distribution_normal';

% --- 分箱法 (Binning Method) ---
    sampling_method_name = '(Binning Method)'; % 定义方法名称
    Rs_ini=normrnd(mu,sigma,1000*N,1);
    Rs_ini = Rs_ini(Rs_ini > 0);
    x_bins=linspace(max(mu-3*sigma,0.1), mu+3*sigma, 20);
    Rs=[];
    for i=1:length(x_bins)-1
       Rs_qu=Rs_ini(Rs_ini >= x_bins(i) & Rs_ini < x_bins(i+1));
       if isempty(Rs_qu), continue; end
       bin_center = (x_bins(i+1) + x_bins(i)) / 2;
       pdf_val = normpdf(bin_center, mu, sigma);
       num=round(N*(x_bins(i+1)-x_bins(i))*pdf_val);
       if num > 0
           random_element = randsample(Rs_qu, num, true);
           Rs=[Rs; random_element];
       end
    end

% ---------------------------- 直接随机采样 --------------------------- %

    % --- 直接随机采样 (Direct Random Sampling) ---
    % sampling_method_name = '(Direct Sampling)'; % 如果使用此方法，请取消本行注释
    % Rs=normrnd(mu,sigma,N,1);

    % --- 样本数量调整 ---
    if length(Rs)>N
        deleteIdx = randsample(1:length(Rs), length(Rs)-N);
        Rs(deleteIdx) = [];
    end
    if length(Rs)<N
        Supplement_element=randsample(Rs_ini, N-length(Rs), true);
        Rs=[Rs; Supplement_element];
    end

% 绘制直方图
    figure('Visible', 'off');
    histogram(Rs, 10, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 0.8);
    hold on; box on;
    x = linspace(min(Rs)*0.8, max(Rs)*1.2, N);
    y = normpdf(x, mu, sigma);
    plot(x, y, 'r-', 'LineWidth', 2);

    % *** 核心修改：动态设置标题 ***
    title(['Normal Distribution ', sampling_method_name]);

    xlabel('Particle Radius');
    ylabel('Probability Density');
    set(gca,'LineWidth',1.5);  set(gca,'FontSize',16);
    legend('Sample Data', 'Theoretical PDF','location', 'northeast', 'LineWidth',0.8);
    hold off;
    Rs = Rs(:);
    saveas(gcf, fullfile(temp_fig_folder, '01_Radius_Distribution.png'));
    drawnow;
    close(gcf); % <-- 新增：保存后关闭，释放内存
end
Rs = Rs(:); % 确保最终Rs是列向量

% ---------------------------- continuous uniform distribution ----------------------------- %
% % a_left=mu-sigma*(3^0.5);
% % b_right=mu+sigma*(3^0.5);
% % Rs_ini=unifrnd(a_left,b_right,1000*N,1);
% % x=linspace(a_left,b_right,21); % 划分间隔
% % Rs=[];
% % for i=1:length(x)-1
% % Rs_qu=Rs_ini(Rs_ini >= x(i) & Rs_ini <= x(i+1)); % 取间隔内数据
% % num=N*(x(i+1)-x(i))* (1 / (b_right - a_left)); % 总数据N乘以bin面积，得到数据频数
% % num=round(num);
% % % num=floor(num);
% % random_element = randsample(Rs_qu, num); % 取间隔内数据
% % random_element=random_element';
% % Rs=[Rs,random_element];
% % end
% %
% % % % % Rs=unifrnd(a_left,b_right,N,1);
% %
% % figure;
% % histogram(Rs, 10, 'Normalization', 'pdf', 'EdgeColor', 'b');
% % hold on;
% %
% % x = linspace(a_left - 0.2, b_right + 0.2, 1000);  % 生成均匀分布的x值
% % pdf_theory = 1 / (b_right - a_left) * ones(size(x));  % 均匀分布的PDF，常数值1/(b_right - a_left)
% %
% % % 绘制理论的均匀分布PDF
% % plot(x, pdf_theory, 'r', 'LineWidth', 2);
% %
% % title(['Uniform Distribution: a_{left} = ', num2str(a_left), ', b_{right} = ', num2str(b_right)]);
% % xlabel('Value');
% % ylabel('Probability Density');
% % legend('Histogram of Rs', 'Theoretical PDF');
% % grid on;
% % hold off;
% % Rs=Rs';

%% 生成球的位置
if (ndm == 2)
	V_total = sum(pi*Rs.^2);
elseif (ndm == 3)
	V_total = sum(4/3*pi*Rs.^3);

end
R0 = (sum(Rs.^ndm)/phi_0)^(1/ndm); % 外圈半径

if ndm==3
    phi=Rs+(R0-2*Rs).*rand(N, 1);
    alpha=pi*rand(N, 1);
    theta=2*pi*rand(N, 1);
    Ps(:,1)=phi.*sin(theta).*sin(alpha);
    Ps(:,2)=phi.*cos(theta).*sin(alpha);
    Ps(:,3)=phi.*cos(alpha);
    Ps(:,4)=Rs;
end
if ndm==2
    phi=Rs+(R0-2*Rs).*rand(N, 1);
    theta=2*pi*rand(N, 1);
    Ps(:,1)=phi.*sin(theta);
    Ps(:,2)=phi.*cos(theta);
    Ps(:,3)=Rs;
end

figure('Visible', 'off'); % 创建一个新的、干净的图窗
title('Initial Particle Placement', 'FontSize', 16); % (可选)给图形加个标题，方便区分
Draw_particle_Cricle(Ps, N, R0, ndm); % 使用正确的四个参数来调用函数
saveas(gcf, fullfile(temp_fig_folder, '02_Initial_Placement.png'));
drawnow;
close(gcf); % <-- 新增：保存后关闭，释放内存

% k=2;
% Draw_particle_period(Ps,N,k+1,L0,ndm);
%---------------------------------
%PiPj = nchoosek(1:N,2);
%RiRj = Ps(PiPj(:,1),ndm+1) + Ps(PiPj(:,2),ndm+1);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% --- 5. RUN QUASI-PHYSICAL PACKING ALGORITHM (已添加迭代监控) ---
disp('开始粒子堆积迭代 ');
k=1;Ps_save{k,1}=Ps;tempN=0;L_change=[];Overlapi_change=[]; % Save history values   i=1
fetmp = [];imove = 0;FP=ones(N,1);F_tot=ones(N,1);ki=0;F_sum2=1;SerialNum=(1:N)';Di=1000*max(Rs);
packing_converged = false;
while 1
    Ps_Virc=Circle_Boundary(R0+1,ndm);
    N_Vir=size(Ps_Virc,1);
    Ps=[Ps;Ps_Virc];
    [PiPj_1]=Contact_Detection_By_Boundary_Box1(Ps,ndm,N+N_Vir);
    RiRj_1 = Ps(PiPj_1(:,1),ndm+1) + Ps(PiPj_1(:,2),ndm+1);

%-----------------------------------------
    Diffij = Ps(PiPj_1(:,1),1:ndm) - Ps(PiPj_1(:,2),1:ndm); % i-j
    Distij = sqrt(sum(Diffij.*Diffij,2));
    [PiPj_f,F,Fij]=calculate_force(PiPj_1,Diffij,Distij,RiRj_1,ndm);

    PiPj_tot =PiPj_f;
    F_tot = F;
    Fij_tot =Fij;

%-----------------------------------------
    FP=full(sparse(PiPj_tot(:,[ones(1,ndm),ones(1,ndm)*2]),ones(size(F_tot,1),1)*[1:ndm,1:ndm],[Fij_tot,-Fij_tot],N+N_Vir,ndm));
    num_PiPj=size(PiPj_f(PiPj_f(:,1)<=N,:),1);
    PiPj_tot (num_PiPj+1:end,:)=0;
    F_tot (num_PiPj+1:end,:)=0;
    Fij_tot(num_PiPj+1:end,:)=0;
    FP(N+1:N+N_Vir,:)=0;

%----------------------------------------------------------------------
    Overlapi = mean(F_tot);
    Overlapi_change = [Overlapi_change;Overlapi];

% --- 【新增】 状态监控 ---
    max_force = max(sqrt(sum(FP.*FP,2))); % 预先计算最大力
    if mod(k, 1000) == 0 || k == 1 % 在第1次和之后每500次迭代时打印
        fprintf('迭代: %6d | 最大力: %.4e / %.2e | 平均重叠: %.4e / %.2e | 边界半径 R0: %.4f\n', ...
                k, max_force, ftol, Overlapi, ttol, R0);
    end

% --- 监控代码结束 ---

    if  max_force < ftol % 使用预先计算好的 max_force
        if  Overlapi<ttol
            Ps(:,1:ndm)=Ps(:,1:ndm)+deltat0*FP; %update Ps
            packing_converged = true;
            disp('收敛条件满足，迭代正常结束。'); % 增加结束提示
            break;
        else
            % k % (可选)可以注释掉这个简单的k值打印
            R0=R0*(1+Overlapi/beta);
        end
    end
    Ps(:,1:ndm)=Ps(:,1:ndm)+deltat0*FP; %update Ps
    Ps=Ps(1:N,:);

%----------------------------------------------------------------------
    L_ite{k,1}=R0;
    k = k + 1;
    if k>Tk
       disp('达到最大迭代次数，循环终止。'); % 增加结束提示
       break;
    end
end

if ~packing_converged
    packing_diagnostic = struct('converged', false, 'iterations', min(k, Tk), ...
        'iteration_limit', Tk, 'maximum_force', max_force, ...
        'force_tolerance', ftol, 'mean_overlap', Overlapi, ...
        'overlap_tolerance', ttol, 'seed', workflow_random_seed, ...
        'particle_count', N, 'radius', R0);
    diagnostic_path = fullfile(pwd, 'packing_failure.mat');
    save(diagnostic_path, 'packing_diagnostic', 'Ps', 'Rs', 'Overlapi_change');
    error('BatteryWorkflow:PackingDidNotConverge', ...
        ['Packing did not satisfy its original force and overlap tolerances ' ...
         'within %d iterations. COMSOL construction has not started. ' ...
         'Inspect the saved diagnostic: %s'], Tk, diagnostic_path);
end

% ---------------------------- save data ----------------------------- %
%Overlapi

if ndm==2
    phi_end1=V_total/(pi*R0^ndm );
else
    phi_end1=V_total/(4/3*pi*R0^ndm );
end
times=toc;
% --- 格式化输出最终结果 ---
fprintf('\n-------------------- 迭代完成 --------------------\n');
fprintf('  最终体积分数 (phi_end1):  %.4f\n', phi_end1);
fprintf('  总耗时:                %.2f 秒\n', times);
fprintf('-----------------------------------------------------------\n');

%% --- 6. (修改) 为COMSOL生成几何与取向数据 ---

% --------------------------------------------------------------------------
% --------------------------------------------------------------------------
disp('正在计算并保存用于COMSOL的几何与取向数据...');
centers = Ps(:, 1:2); % 获取最终的晶粒中心
[V, C] = voronoin(centers); % V是顶点坐标, C是顶点索引

% --- 关键：过滤掉无限单元格 ---
valid_cell_indices = [];
for i = 1:length(C)
    if all(C{i} ~= 1) % 检查是否包含无限顶点(索引为1)
        valid_cell_indices = [valid_cell_indices; i];
    end
end
C_finite = C(valid_cell_indices); % 只保留有限的单元
Ps_finite = Ps(valid_cell_indices, :); % 只保留有限单元对应的粒子信息
N_finite = length(C_finite);

% --- (可选但推荐) 为每个晶粒单独存储顶点 ---
polygons = cell(N_finite, 1);
for i = 1:N_finite
    polygons{i} = V(C_finite{i}, :);
end

fprintf('-----------------------------------------------------------\n');

fprintf('-----------------------------------------------------------\n');
% --- 【【【【 代码修改：按论文 Fig 3a 计算 alpha, theta, beta 】】】】 ---
disp('正在为每个晶粒计算 alpha, theta, beta 角...');

centers_finite = Ps_finite(:, 1:2); % Ps_finite 已在第 338 行定义

% 步骤 1: 计算 alpha (径向取向角)
% alpha = atan2(y, x)
alpha_vec = atan2(centers_finite(:, 2), centers_finite(:, 1));

fprintf('-----------------------------------------------------------\n');
fprintf('-----------------------------------------------------------\n');

% 步骤 2: 生成 theta (取向差)
% --- 【【【【 代码修改：切换为 "完全随机" 取向 (v4) 】】】】 ---
disp('现在为 "完全随机" 取向 (Random Orientations)...');

% 步骤 2: 生成 theta (取向差)
% 根据论文 [Fig 3(a), source 256]，取向差角 theta 的范围是 [0, pi/2]。
% 我们现在使用均匀随机分布来替代 EBSD 拟合的 "拒绝取样法"。

theta_max = pi/2;
theta_vec = rand(N_finite, 1) * theta_max; % <-- 这是新的核心代码

disp('... "完全随机" 的 theta_vec (0-pi/2) 已生成。');

fprintf('-----------------------------------------------------------\n');

% --- 【【【【 代码修改：切换为 "拒绝采样法" 取向 】】】】 ---
% 【【【【 科学复现 (v3)：精确拟合的 Rejection Sampling 】】】】
% 我们使用一个精确的模型 P(theta) = A*exp(-k*theta) + C
% 来匹配论文 Fig 3c 的红色 "Fitted Curve"

% 1. 定义我们精确拟合的目标 PDF
%    根据数据点 P(1.0)≈0.5, P(1.5)≈0.4 拟合:
%    P(theta) = 0.9 * exp(-1.5 * theta) + 0.3
%A = 0.9;
%k = 1.5;
%C = 0.3;
%target_pdf = @(theta) A * exp(-k * theta) + C;

% 2. 定义 Proposal 分布 (我们用一个简单的均匀分布)
%theta_max = pi/2; % 论文定义的范围 [0, pi/2]
%prob_max = target_pdf(0); % P(theta) 的最大值在 theta=0 处 (1.2)

% 3. 使用 Rejection Sampling 填充 theta_vec
%theta_vec = zeros(N_finite, 1); % 预分配向量
%count = 0;
%while count < N_finite
    % 步骤 a: "提议"一个随机点 (在 0 到 pi/2 的矩形框内)
    %theta_try = rand() * theta_max;  % 随机 "x"
    %prob_try  = rand() * prob_max;  % 随机 "y"

    % 步骤 b: "接受" 这个点，如果它在我们的红色曲线下方
   % prob_accept = target_pdf(theta_try);

    %if prob_try <= prob_accept
        % 接受这个样本
        %count = count + 1;
        %theta_vec(count) = theta_try;
    %end
    % 如果 prob_try > prob_accept，则 "拒绝" 该样本，循环继续
%end

% --- 【【【【 代码修改：切换为 "拒绝采样法" 取向 】】】】 ---

% --- 【【【【 科学复现代码结束 】】】】 ---
% 步骤 3: 计算 beta (晶体取向角)
% 根据论文公式 (Fig 3a): beta = alpha + theta
beta_vec = alpha_vec + theta_vec;

% 步骤 4: 将角度保存到 Ps_finite 矩阵中
% 第4列存 beta (用于COMSOL)
% 第5列存 theta (用于我们画图)
Ps_finite = [Ps_finite, beta_vec, theta_vec];

% ------------------- 修正开始 -------------------
% 格式化 R0 值，保留两位小数
R0_str = sprintf('R%.2f', R0);

% 构造新的文件名：将 R0 嵌入文件名中
% output_filename_txt = ['grain_orientations_', R0_str, '.txt'];
% ------------------- 修正结束 -------------------

% 我们提取 x(第1列), y(第2列), beta(第4列)
% writematrix(Ps_finite(:, [1, 2, 4]), output_filename_txt);
% fprintf('已成功导出晶粒取向文件：%s\n', output_filename_txt);

fprintf('-----------------------------------------------------------\n');
% --- 【【【【 代码修改结束 】】】】 ---

% --- 步骤 7: (更新) 保存为 .mat 文件 ---
output_filename_mat = 'geometry_for_comsol.mat'; % 保留你原来的文件名
save(output_filename_mat, 'V', 'C_finite', 'polygons', 'Ps_finite', 'R0');
fprintf('已更新并成功为COMSOL生成几何文件：%s\n', output_filename_mat);
fprintf('-----------------------------------------------------------\n');
% --------------------------------------------------------------------------
% ----------------- 【【【【 代码修改区域结束 】】】】 -----------------
% --------------------------------------------------------------------------

% ---------------------------- save data ----------------------------- %
figure('Visible', 'off'); % 创建一个新的、干净的图窗
title('Final Particle Placement', 'FontSize', 16); % (可选)给图形加个标题，方便区分
Draw_particle_Cricle(Ps, N, R0, ndm); % 使用正确的四个参数来调用函数
saveas(gcf, fullfile(temp_fig_folder, '03_Final_Placement.png'));
drawnow;
close(gcf); % <-- 新增：保存后关闭，释放内存

%% --- 10. VISUALIZE FINAL POLYCRYSTAL STRUCTURE ---
Draw_Polycrystal(Ps, R0);% 调用我们刚刚创建的新函数
saveas(gcf, fullfile(temp_fig_folder, '04_Voronoi_Uncolored.png'));
close(gcf); % <-- 新增：保存后关闭，释放内存

fprintf('-----------------------------------------------------------\n');

%% --- 11. (V2 修正) 可视化 (按 Beta 角着色) ---
disp('正在生成按晶体取向角 beta (折叠) 着色的多晶图...'); % <-- 修正1: 更改提示

figure('Visible', 'off');
hold on;

% --- (for 循环 patch 绘图部分) ---
beta_angles = Ps_finite(:, 4); % <-- 修正2: 读取 beta (第4列), 而不是 theta (第5列)

for i = 1:N_finite
    vert_indices = C_finite{i};
    poly_verts = V(vert_indices, :);
    beta_angle = beta_angles(i); % <-- 修正3: 使用 beta

% --- 修正4: 插入学长参考代码中的 "折叠" 逻辑 ---
% 我们需要将 beta (范围可能是 [-pi, pi] + [0, pi/2])
% 首先转换到 [0, 2*pi] 范围 (0~360度)
beta_angle_0_2pi = mod(beta_angle, 2*pi);

% 然后应用 [0, pi/2] 的折叠 (0~90度)
if  beta_angle_0_2pi >= 0 && beta_angle_0_2pi < pi/2
    % 1. 角度在第一象限 (0~90度), 已经是锐角, 直接使用
    val = beta_angle_0_2pi;

elseif beta_angle_0_2pi >= pi/2 && beta_angle_0_2pi < pi
    % 2. 角度在第二象限 (90~180度), 例如 120 度。
    %    我们取它与 "负x轴" (pi) 的夹角： pi - 120 = 60 度。
    val = pi - beta_angle_0_2pi;

elseif beta_angle_0_2pi >= pi && beta_angle_0_2pi < 3*pi/2
    % 3. 角度在第三象限 (180~270度), 例如 210 度。
    %    我们取它与 "负x轴" (pi) 的夹角： 210 - pi = 30 度。
    val = beta_angle_0_2pi - pi;

else % 范围 [3*pi/2, 2*pi]
    % 4. 角度在第四象限 (270~360度), 例如 300 度。
    %    我们取它与 "正x轴" (2*pi) 的夹角： 2*pi - 300 = 60 度。
    val = 2*pi - beta_angle_0_2pi;
end
% --- 修正4结束 ---

patch(poly_verts(:, 1), poly_verts(:, 2), val, 'EdgeColor', 'k'); % <-- 修正5: 使用 val 绘图
end
% --- (for 循环结束) ---

% --- 【【【【 关键：从 (Random_orientation_distribution.m) 移植的色彩映射表 】】】】 ---
%

% 1. 定义关键颜色的 RGB 值 (0-255)
cmap = [ ...
    235, 42,   114;
   245, 85,   121;
    255, 132,  158;    % 粉色系
    254, 148,   62;   % 橙色系
   252, 183,   78;    % 亮黄色
   254, 222,  160;
    192, 255,   0;    % 柠檬绿
      0, 255,   0;    % 鲜绿色
      0, 255, 255;    % 浅青色
    160, 186, 255;    % 天蓝色
    100,  92, 254;
    112,  26, 251;     % 蓝色系
]/255; %

% 2. 定义插值点
x_pops = linspace(0, 1, size(cmap,1)); %
xq = linspace(0, 1, 256); %

% 3. 生成插值后的 colormap
cmap_interp = [
    interp1(x_pops, cmap(:,1), xq)
    interp1(x_pops, cmap(:,2), xq)
    interp1(x_pops, cmap(:,3), xq)
]'; %

% 4. 应用这个自定义的 colormap
colormap(flipud(cmap_interp)); %
% --- 【【【【 自定义 Colormap 结束 】】】】 ---

% 【【【【 Colorbar 和 顶部 Theta 标签 (保持不变) 】】】】
%%cb = colorbar; %

% 1. 将颜色范围固定为 [0, pi/2] (约 1.57)
caxis([0, pi/2]); %

% 2. 设置刻度
%%set(cb, 'Ticks', [0, 0.5, 1.0, 1.5]); %
%%set(cb, 'FontSize', 14); %

% 3. (关键) 将标签 '\beta' 添加到顶部
%%cb.Label.String = '\beta'; %
%%cb.Label.FontSize = 16; %
%%cb.Label.Rotation = 0; %
%%cb.Label.VerticalAlignment = 'bottom'; %
%%cb.Label.HorizontalAlignment = 'center'; %
% --- 【【【【 Colorbar 结束 】】】】 ---

% 设定坐标轴 (保持不变)
axis equal; %
axis tight; %
xlim([-R0, R0]); %
ylim([-R0, R0]); %

% --- 【【【【 【【【【 你的修改在这里 】】】】 】】】】 ---
axis off; %
%%title('Final Polycrystal Microstructure (Colored by \beta)', 'FontSize', 16); % 【已修正】
% --- 【【【【 【【【【 修改结束 】】】】 】】】】 ---

% --- 【【【【 核心修改：强制 512x512 输出 】】】】 ---
% 1. 让图像填满画布
set(gca, 'Position', [0 0 1 1], 'Units', 'normalized');

% 2. 定义文件路径
save_path = fullfile(temp_fig_folder, '05_Voronoi_Theta_Colored.png');

% 3. 先导出高清图 (使用 gcf)
exportgraphics(gcf, save_path, 'Resolution', 300);  % <-- 【已修复】把 f 改成了 gcf

% 4. 再读取并缩放到 512x512
try
    im_high = imread(save_path);
    im_512 = imresize(im_high, [512, 512]); % 简单缩放
    imwrite(im_512, save_path); % 覆盖保存
catch orientation_error
    fprintf(2, 'Required 512-by-512 orientation export failed: %s\n', save_path);
    rethrow(orientation_error);
end

% --- 【【【【 修改结束 】】】】 ---
hold off;
drawnow;
close(gcf); % <-- 新增：保存后关闭，释放内存

fprintf('根据晶体取向角 (Beta) 着色的图生成完毕 (像素已设为512x512)。\n');
fprintf('-----------------------------------------------------------\n');
fprintf('-----------------------------------------------------------\n');
fprintf('-----------------------------------------------------------\n');
