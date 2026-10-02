% =========================================================================
% ===       COMSOL 建模脚本: build_full_comsol_model.m（融合版）       ===
% =========================================================================
%
%   本脚本执行：
%   1. 加载你的 'geometry_for_comsol.mat'
%   2. 使用你的逻辑（Polygon, Union）从头构建 COMSOL 几何
%   3. 使用你的逻辑（Ps_finite）分配晶体角度和物理场
%   4. 【新增】使用“学长”的设置为模型划分“超细”网格
%   5. 保存并打开模型
%
% =========================================================================

%clc;
clear model;
%close all; % 关闭所有旧的 figure 窗口
disp('--- COMSOL 建模脚本已启动 ---');

% --- 0. Connect to the existing COMSOL session without deleting its models. ---
import com.comsol.model.*
import com.comsol.model.util.*

try
    % --- 1. 加载来自 main_D21origin.m 的数据 ---
    disp('加载 geometry_for_comsol.mat ...');
    load('geometry_for_comsol.mat'); % 加载: V, C_finite, polygons, Ps_finite, R0

    N_finite = length(polygons); % 获取晶粒数
    r0_value = R0; % 获取外径

    fprintf('成功加载 %d 个晶粒的数据。\n', N_finite);

    fprintf('------------------------------------------------\n');

    % --- 2. 创建模型和物理场（来自你的脚本） ---
    disp('创建 COMSOL 模型、参数和物理场接口...');
    model = ModelUtil.createUnique('Battery');
    model.modelPath(pwd); % Use the isolated working directory.
    model.component.create('comp1', true);
    model.component('comp1').geom.create('geom1', 2);
    model.component('comp1').geom('geom1').lengthUnit('um');

    % （你的参数）
    model.param.group.create('par2');
    model.param('par2').set('Omega11', '7.3098e-7[m^3/mol]');
    model.param('par2').set('Omega33', '-1.1049e-6[m^3/mol]');
    model.param('par2').set('Mm', '0.09728[kg/mol]');
    model.param('par2').set('D11', '7e-15[m^2/s]');
    model.param('par2').set('D33', '7e-16[m^2/s]');
    model.param('par2').set('cs_max', '50060[mol/m^3]');
    model.param('par2').set('cs_0', '0.9*cs_max');

    % -----------------------------v --- -----------------------------v ---    -----------------------------v ---
    % -----------------------------v --- -----------------------------v ---    -----------------------------v ---
    % -----------------------------v --- -----------------------------v ---    -----------------------------v ---
    % --   % 【注意：此处修改 c_rate】 ----------------------------------v ---
    % -------   % 【注意：此处修改 c_rate】 ------------------------------v ---
    % --------   % 【注意：此处修改 c_rate】 -----------------------------v ---
    % ----   % 【注意：此处修改 c_rate】 --------------------------------v ---
    % 【注意：此处修改 c_rate】

    model.param('par2').set('c_rate', '4');

    % --   % 【注意：此处修改 c_rate】 ----------------------------------v ---
    % --   % 【注意：此处修改 c_rate】 ----------------------------------v ---
    % --   % 【注意：此处修改 c_rate】 ----------------------------------v ---
    % ------------------------------v ---

    % -----------------------------v --- -----------------------------v ---    -----------------------------v ---
    % -----------------------------v --- -----------------------------v ---    -----------------------------v ---
    % -----------------------------v --- -----------------------------v ---    -----------------------------v ---

    model.param('par2').set('rp', string(r0_value)+'[um]');
    model.param('par2').set('V', '(4/3)*pi*rp^3');
    model.param('par2').set('S', '4*pi*rp^2');
    model.param('par2').set('J', '(V/S)*0.6*cs_max*c_rate/3600[s]');
    model.param('par2').label('Lithium ion diffusion');

    model.param.set('E', '140[GPa]');
    model.param.set('v', '0.3');
    model.param.set('rh0', '4.87[g/cm^3]');
    model.param.set('C11', '259[GPa]');
    model.param.set('C12', '75[GPa]');
    model.param.set('C22', '194[GPa]');
    model.param.set('C44', '76[GPa]');
    model.param.label('mechanics');

    % （你的物理场）
    model.component('comp1').physics.create('tds', 'DilutedSpecies', 'geom1');
    model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
    model.component('comp1').physics('solid').create('rms1', 'RigidMotionSuppression', 2);
    model.component('comp1').mesh.create('mesh1');
    model.study.create('std1');
    model.study('std1').create('time', 'Transient');
    model.study('std1').feature('time').set('tlist', 'range(0,100,2400)');

    fprintf('------------------------------------------------\n');

    % --- 3. 【第 1 轮循环】（只构建几何）（来自你的脚本） ---
    fprintf('正在为 %d 个晶粒构建几何...\n', N_finite);
    polygon_tags = {};
    domain_indices = [];

    for i = 1:N_finite
        vertices = polygons{i};
        if size(vertices, 1) < 3
            fprintf('警告：第 %d 个多边形顶点数过少，已跳过。\n', i);
            fprintf('------------------------------------------------\n');
            continue;
        end

        poly_tag = ['p' num2str(i)];
        polygon_tags{end+1} = poly_tag;
        domain_indices(end+1) = i;

        model.component('comp1').geom('geom1').feature.create(poly_tag, 'Polygon');
        model.component('comp1').geom('geom1').feature(poly_tag).set('x', vertices(:,1)');
        model.component('comp1').geom('geom1').feature(poly_tag).set('y', vertices(:,2)');
    end

    fprintf('-----------------------------------------------------------\n');

    % --- 4. 【构建并敲定几何】（来自你的脚本） ---
    disp('正在合并晶粒并进行圆形裁剪...');
    model.component('comp1').geom('geom1').feature.create('union1', 'Union');
    model.component('comp1').geom('geom1').feature('union1').selection('input').set(polygon_tags);
    model.component('comp1').geom('geom1').feature('union1').set('intbnd', 'on');

    model.component('comp1').geom('geom1').feature.create('circ1', 'Circle');
    model.component('comp1').geom('geom1').feature('circ1').set('r', r0_value);

    model.component('comp1').geom('geom1').feature.create('int1', 'Intersection');
    model.component('comp1').geom('geom1').feature('int1').selection('input').set({'union1', 'circ1'});

    fprintf('------------------------------------------------\n');
    disp('已构建最终几何，正在执行 Geom1.run() ...');
    model.component('comp1').geom('geom1').run();

    fprintf('------------------------------------------------\n');

    % --- 5. 【第 2 轮循环】（分配物理场）（来自你的脚本） ---
    fprintf('正在为 %d 个有效晶粒分配晶体力学与物理场属性...\n', length(domain_indices));
    Omega11_val = 7.3098e-7;
    Omega33_val = -1.1049e-6;
    Mm_val = 0.09728;
    domain_counter = 1;

    for i = domain_indices

        % --- a. 提取【你自己的】beta 角 ---
        beta_angle = Ps_finite(i, 4);

        % --- b. 创建选区 ---
        sel_tag = ['sel' num2str(i+1)];
        model.component('comp1').selection.create(sel_tag, 'Explicit');
        model.component('comp1').selection(sel_tag).label(['Grain ' num2str(i)]);
        model.component('comp1').selection(sel_tag).set(domain_counter);

        % --- c. 设置旋转坐标系 ---
        sys_tag = ['sys' num2str(i+1)];
        model.component('comp1').coordSystem.create(sys_tag, 'Rotated');
        model.component('comp1').coordSystem(sys_tag).set('inPlaneAngle', string(beta_angle));

        % --- d. 分配扩散物质传输（tds） ---
        tds_tag = ['cdm' num2str(i+1)];
        model.component('comp1').physics('tds').create(tds_tag, 'ConvectionDiffusionMigration', 2);
        model.component('comp1').physics('tds').feature(tds_tag).selection.named(sel_tag);
        model.component('comp1').physics('tds').feature(tds_tag).set('D_c', {'D11'; '0'; '0'; '0'; 'D33'; '0'; '0'; '0'; '0'});
        model.component('comp1').physics('tds').feature(tds_tag).set('coordinateSystem', sys_tag);

        % --- e. 分配固体力学（solid） ---
        solid_tag = ['lemm' num2str(i+1)];
        model.component('comp1').physics('solid').create(solid_tag, 'LinearElasticModel', 2);
        model.component('comp1').physics('solid').feature(solid_tag).selection.named(sel_tag);
        model.component('comp1').physics('solid').feature(solid_tag).set('SolidModel', 'Anisotropic');
        model.component('comp1').physics('solid').feature(solid_tag).set('AnisotropicOption', 'AnisotropicVo');
        model.component('comp1').physics('solid').feature(solid_tag).set('DVo_mat', 'userdef');

        % --- 使用你脚本中的原始 6x6 矩阵 ---
        model.component('comp1').physics('solid').feature(solid_tag).set('DVo', ...
        {'C11'; 'C12'; '0'; '0'; '0'; '0';
        'C12'; 'C22'; '0'; '0'; '0'; '0';
        '0';  '0';  '0';  '0';  '0';  '0';
        '0';  '0';  '0';  'C44';  '0'; '0';
        '0';  '0';  '0';  '0';  '0';  '0';
        '0';  '0';  '0';  '0';  '0';  '0'});

        model.component('comp1').physics('solid').feature(solid_tag).set('rho_mat', 'userdef');
        model.component('comp1').physics('solid').feature(solid_tag).set('rho', 'rh0');
        model.component('comp1').physics('solid').feature(solid_tag).set('coordinateSystem', sys_tag);

        % --- f. 分配多物理场耦合（HygroscopicSwelling） ---
        hs_tag = ['hs' num2str(i)];

        % --- 使用你脚本中的原始 9x1 张量 ---
        Omega_now11 = Omega11_val*(cos(beta_angle))^2 + Omega33_val*(sin(beta_angle))^2;
        Omega_now33 = Omega11_val*(sin(beta_angle))^2 + Omega33_val*(cos(beta_angle))^2;
        Omega_now13 = -cos(beta_angle)*sin(beta_angle)*Omega11_val + cos(beta_angle)*sin(beta_angle)*Omega33_val;
        Omega_now = [Omega_now11, Omega_now13; Omega_now13, Omega_now33];
        beta_now = Omega_now / Mm_val;
        beta_0_dim3 = zeros(3,3);
        beta_0_dim3(1:2,1:2) = beta_now;
        beta_tensor = reshape(beta_0_dim3, 9, 1);

        model.component('comp1').multiphysics.create(hs_tag, 'HygroscopicSwelling', 2);
        model.component('comp1').multiphysics(hs_tag).selection.named(sel_tag);
        model.component('comp1').multiphysics(hs_tag).set('Mm', 'Mm');
        model.component('comp1').multiphysics(hs_tag).set('beta_h_mat', 'userdef');
        model.component('comp1').multiphysics(hs_tag).set('beta_h', beta_tensor);

        domain_counter = domain_counter + 1;
    end

    fprintf('%d 个晶粒的物理场分配完毕。\n', length(domain_indices));

    fprintf('------------------------------------------------\n');
    fprintf('------------------------------------------------\n');

    % --- 6. 设置全局物理场（来自你的脚本） ---
    disp('设置全局边界条件和初始值...');
    model.component('comp1').physics('tds').create('fl1', 'FluxBoundary', 1);
    model.component('comp1').physics('tds').feature('fl1').selection.all;
    model.component('comp1').physics('tds').feature('fl1').set('species', true);
    model.component('comp1').physics('tds').feature('fl1').set('J0', '-J');
    model.component('comp1').physics('tds').feature('init1').set('initc', 'cs_0');

    fprintf('------------------------------------------------\n');
    fprintf('------------------------------------------------\n');

    % =====================================================================
    disp('正在划分“超细”网格...');
    % 'autoMeshSize(2)' 对应“超细”（Extra fine）
    model.component('comp1').mesh('mesh1').autoMeshSize(2);

    % =====================================================================
    % --- 新增功能结束 ---
    % =====================================================================

    fprintf('------------------------------------------------\n');

    % --- 7. 先保存，再打开（来自你的脚本） ---
    output_model_file = 'comsol_MERGED_SCIENTIFIC_model.mph'; % 新文件名
    disp('正在保存模型...');
    fprintf('------------------------------------------------\n');

    mphsave(model, output_model_file); % <-- 1. 先保存

     %disp('正在打开 COMSOL 窗口...');
     %mphlaunch(model); % <-- 2. 再打开
     %fprintf('--- 成功！模型已保存到 %s ---\n', output_model_file);
     %disp('这是一个包含完整几何和物理场的模型，已进行网格划分，现在直接计算即可。');

catch e
    disp('--- 发生错误 ---');
    disp(e.message);
    if ~isempty(e.stack)
        fprintf('错误发生在文件 %s 的第 %d 行\n', e.stack(1).file, e.stack(1).line);
    end
    disp('请确认 COMSOL Livelink 已启动，并且所有路径都正确。');
    rethrow(e); % Do not continue to export an incomplete model.
end
