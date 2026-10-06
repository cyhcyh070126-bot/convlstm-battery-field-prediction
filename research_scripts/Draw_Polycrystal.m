function Draw_Polycrystal(Ps, R0)
% --- Draw_Polycrystal (论文复现最终版) ---
% 目的: 严格按照论文 Fig. 1/2 的要求，生成一个无尖刺的圆形多晶图。
% 特点:
% 1. 内置 figure 和 title 创建。
% 2. 修正了 colormap，确保 "红(低) -> 蓝(高)" 的论文配色。
% 3. 包含了无限单元格过滤，确保无"尖刺"。
% 4. 包含了圆形裁剪遮罩，确保边界圆滑。

% --- 1. 创建图窗和标题 ---
figure('Visible', 'off'); % % 创建一个新图窗
ax = gca; % 获取当前坐标轴
hold(ax, 'on');
title(ax, 'Final Polycrystal Microstructure', 'FontSize', 16, 'FontWeight', 'bold');
axis equal; 
box on; 
set(gcf, 'color', 'w'); % 设置背景为白色

% --- 2. 准备数据和颜色映射 ---
centers = Ps(:, 1:2);
radii = Ps(:, 3);
N = size(Ps, 1);
if N < 3, warning('粒子数量过少，无法绘制有效的 Voronoi 图。'); return; end

[V, C] = voronoin(centers); 

% 【核心修正】：修正颜色映射
% 论文中的配色是 "Red-Yellow-Green-Blue" (低-高)
% 默认 'jet' 是 "Blue-Green-Yellow-Red" (低-高)
% 因此，我们需要翻转(flipud) 'jet' colormap
colors_map = flipud(jet(256)); 
colormap(ax, colors_map); % 将修正后的 colormap 应用于坐标轴

% --- 3. 计算颜色索引 (使用固定的论文范围) ---
min_r_fixed = 1.2; % 固定的颜色范围下限
max_r_fixed = 2.6; % 固定的颜色范围上限

% 将半径钳位(clamp)到固定范围内，确保颜色显示一致
clamped_radii = max(min(radii, max_r_fixed), min_r_fixed);

if abs(max_r_fixed - min_r_fixed) < eps
    color_indices = 128 * ones(N, 1); % 避免除零
else
    % 归一化到 0-255 并加 1，得到 1-256 的索引
    color_indices = round(((clamped_radii - min_r_fixed) / (max_r_fixed - min_r_fixed)) * 255) + 1;
end
color_indices = max(1, min(256, color_indices)); % 再次确保索引在 1-256 范围内

% --- 4. 【核心绘图】过滤无限单元格并绘图 ---
for i = 1:N 
    if all(C{i} ~= 1) % 关键一步：过滤掉指向 "无限" 顶点的单元
        polygon_vertices = V(C{i}, :);
        fill_color = colors_map(color_indices(i), :);
        % 在指定坐标轴(ax)上绘图
        patch(ax, polygon_vertices(:, 1), polygon_vertices(:, 2), fill_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
    end
end

% --- 5. 创建圆形裁剪遮罩 ---
theta = linspace(0, 2*pi, 200);
axis_limit = R0 * 1.1; % 确保白色遮罩足够大
% 外部方框
outer_vertices_x = [axis_limit, -axis_limit, -axis_limit, axis_limit, axis_limit];
outer_vertices_y = [axis_limit, axis_limit, -axis_limit, -axis_limit, axis_limit];
% 内部圆形 (反向绘制以形成"洞")
inner_hole_x = R0 * cos(fliplr(theta));
inner_hole_y = R0 * sin(fliplr(theta));
% 使用 patch 在 R0 之外创建白色遮罩
patch(ax, [outer_vertices_x NaN inner_hole_x], [outer_vertices_y NaN inner_hole_y], 'w', 'EdgeColor', 'none');

% --- 6. 美化与关闭坐标轴 ---
axis(ax, 'off');
axis(ax, 'tight');

% --- 7. 定制化的颜色条 (与论文一致) ---
cb = colorbar(ax, 'Location', 'southoutside');
fixed_color_range = [1.2, 2.6];
fixed_ticks = [1.2, 1.9, 2.6];
caxis(ax, fixed_color_range); % 强制颜色条使用固定范围
set(cb, 'Ticks', fixed_ticks, 'FontName', 'Times New Roman', 'FontSize', 11);
cb.Label.String = 'Radius (\mum)';
cb.Label.FontSize = 12;
cb.Label.FontName = 'Times New Roman';

hold(ax, 'off');

end