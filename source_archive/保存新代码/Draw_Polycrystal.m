function Draw_Polycrystal(Ps, R0, target_N)
% --- Draw_Polycrystal (晶体棱角增强版) ---
% 1. 核心调整：大幅减少幽灵粒子数量 (180 -> 32)，创造长直的自然晶界。
% 2. 视觉效果：去除了"圆润感"，边缘像切割的宝石，更接近真实多晶。
% 3. 稳定性：保留了 target_N 数据清洗功能，防止红圈复发。

% --- 1. 数据清洗 ---
if nargin >= 3 && ~isempty(target_N)
    if size(Ps, 1) > target_N
        Ps = Ps(1:target_N, :);
    end
end

% --- 2. 创建图窗 ---
figure('Visible', 'off'); 
ax = gca; 
hold(ax, 'on');
title(ax, 'Final Polycrystal Microstructure', 'FontSize', 16, 'FontWeight', 'bold');
axis equal; 
box on; 
set(gcf, 'color', 'w'); 

% --- 3. 准备数据 ---
real_centers = Ps(:, 1:2);
real_radii = Ps(:, 3);
N = size(Ps, 1); 

% --- 4. 【核心修改】稀疏幽灵粒子 (Sparse Ghost Particles) ---
% 之前的 180 个太密了，导致边缘变成圆弧。
% 现在改为 30-40 个，让边缘呈现自然的"多边形切割感"。
N_ghost = 32; 

% 稍微加一点随机扰动，让边界不要太死板
theta_ghost = linspace(0, 2*pi, N_ghost+1)';
theta_ghost(end) = []; 

% 半径设为 1.2 倍 R0，给内部晶粒一点向外延伸的空间，形成尖角
R_ghost_base = R0 * 1.2; 
% 引入微小的随机距离变化 (0% ~ 5%)，增加自然感
random_jitter = 0.05 * rand(size(theta_ghost)); 
current_R = R_ghost_base * (1 + random_jitter);

ghost_centers = [current_R .* cos(theta_ghost), current_R .* sin(theta_ghost)];
all_centers = [real_centers; ghost_centers];

% 计算 Voronoi
[V, C] = voronoin(all_centers); 

% --- 5. 自动颜色映射 ---
colors_map = flipud(jet(256)); 
colormap(ax, colors_map); 

min_val = min(real_radii);
max_val = max(real_radii);
if max_val - min_val < 1e-6, max_val = min_val + 0.1; end

color_indices = round(((real_radii - min_val) / (max_val - min_val)) * 255) + 1;
color_indices = max(1, min(256, color_indices));

% --- 6. 绘图 ---
for i = 1:N 
    v_indices = C{i};
    if all(v_indices ~= 1)
        poly_verts = V(v_indices, :);
        fill_color = colors_map(color_indices(i), :);
        patch(ax, poly_verts(:, 1), poly_verts(:, 2), fill_color, ...
              'EdgeColor', 'k', 'LineWidth', 0.5);
    end
end

% --- 7. 收尾 ---
axis(ax, 'off');
axis(ax, 'tight');
% 视野适应新的边界
xlim([-R0*1.25, R0*1.25]);
ylim([-R0*1.25, R0*1.25]);

cb = colorbar(ax, 'Location', 'southoutside');
caxis(ax, [min_val, max_val]); 
set(cb, 'FontName', 'Times New Roman', 'FontSize', 11);
cb.Label.String = 'Radius (\mum)';
cb.Label.FontSize = 12;
cb.Label.FontName = 'Times New Roman';

hold(ax, 'off');
end