% --- create_comsol_geometry.m (最终修正加强版) ---
%
% 目的：使用 COMSOL LiveLink for MATLAB 自动构建多晶几何模型。
% 修正
% 1. 直接向 COMSOL 传递数值数组，而不是字符串，解决 "Wrong number of arguments" 错误。
% 2. 增加对不合格多边形（顶点数 < 3）的跳过处理，增强脚本的稳健性。
%
clc;
clear model; % 清除可能存在的旧模型

disp('--- 自动建模脚本已启动 (最终修正版) ---');

try
    % --- 1. 加载几何数据 ---
    disp('加载 geometry_for_comsol.mat ...');
    load('geometry_for_comsol.mat');
    N_polygons = length(polygons);
    



    % --- 2. 导入 COMSOL 库 ---
    disp('导入 COMSOL 库...');
    import com.comsol.model.*
    import com.comsol.model.util.*
    



    % --- 3. 创建模型和 2D 几何 ---
    model = ModelUtil.create('PolycrystalModel');
    model.modelNode.create('comp1');
    model.geom.create('geom1', 2);
    



    % --- 4. 循环创建所有多边形 (晶粒) ---
    fprintf('正在创建 %d 个多边形晶粒...\n', N_polygons);
    polygon_tags = {}; % 使用动态元胞数组
    
    for i = 1:N_polygons
        vertices = polygons{i}; % 提取第 i 个多边形的顶点
        
        % 【核心修正2】增加稳健性检查：多边形至少需要3个顶点
        if size(vertices, 1) < 3
            fprintf('警告：第 %d 个多边形顶点数少于3，已跳过。\n', i);
            continue; % 跳过这个无效的多边形
        end
        

        poly_tag = ['p' num2str(i)];
        
        % 将标签添加到列表中
        polygon_tags{end+1} = poly_tag;
        
        model.geom('geom1').feature.create(poly_tag, 'Polygon');
        
        % 【核心修正1】直接传递转置后的数值向量，而不是字符串！
        model.geom('geom1').feature(poly_tag).set('x', vertices(:,1)');
        model.geom('geom1').feature(poly_tag).set('y', vertices(:,2)');
    end
    
    fprintf('... %d 个有效晶粒创建完毕。\n', length(polygon_tags));

    % --- 5. 将所有多边形合并为一个对象 ---
    disp('正在合并所有晶粒...');
    model.geom('geom1').feature.create('union1', 'Union');
    model.geom('geom1').feature('union1').selection('input').set(polygon_tags);
    model.geom('geom1').feature('union1').set('intbnd', 'on'); % 保留内部边界(晶界)
    
    % --- 6. 创建圆形边界并执行裁剪 ---
    disp('正在创建圆形边界并裁剪...');
    model.geom('geom1').feature.create('circ1', 'Circle');
    model.geom('geom1').feature('circ1').set('r', R0);
    
    model.geom('geom1').feature.create('int1', 'Intersection');
    model.geom('geom1').feature('int1').selection('input').set({'union1', 'circ1'});
    
    % --- 7. 构建几何并保存模型 ---
    disp('构建最终几何...');
    model.geom('geom1').run();
    
    mphlaunch(model); % 打开COMSOL图形界面来显示结果
    
    output_model_file = 'comsol_geometry_model.mph';
    mphsave(model, output_model_file);
    
    fprintf('--- 成功！模型已保存为 %s ---\n', output_model_file);
    disp('请在 COMSOL 窗口查看生成的几何模型。');

catch e
    disp('--- 发生错误 ---');
    disp(e.message);
    fprintf('错误发生在文件 %s 的第 %d 行\n', e.stack(1).file, e.stack(1).line);
    disp('请确保所有设置正确。');
end