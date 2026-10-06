function Draw_Polycrystal(Ps, R0)
% --- Draw_Polycrystal (最终决定版 V2) ---
% 目的: 严格按照论文 Fig. 1/2 的要求，生成一个无尖刺的圆形多晶图。
figure; 
hold on;
axis equal; 
box on; 
centers = Ps(:, 1:2);
radii = Ps(:, 3);
N = size(Ps, 1);
if N < 3, warning('粒子数量过少，无法绘制有效的 Voronoi 图。'); return; end
[V, C] = voronoin(centers); 
colors_map = jet(256); 
colormap(gca, colors_map);
min_r = 1.2; 
max_r = 2.6; 
clamped_radii = max(min(radii, max_r), min_r);
if abs(max_r - min_r) < eps
    color_indices = 128 * ones(N, 1);
else
    color_indices = round(((clamped_radii - min_r) / (max_r - min_r)) * 255) + 1;
end
color_indices = max(1, min(256, color_indices)); 

% 4. 【核心修正】过滤无限单元格并绘图
for i = 1:N 
    if all(C{i} ~= 1) 
        polygon_vertices = V(C{i}, :);
        fill_color = colors_map(color_indices(i), :);
        patch(polygon_vertices(:, 1), polygon_vertices(:, 2), fill_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
    end
end

% --- 4. 裁剪视图 (可选，但能让边界更圆滑) ---
theta = linspace(0, 2*pi, 200);
axis_limit = R0 * 1.1;
outer_vertices_x = [axis_limit, -axis_limit, -axis_limit, axis_limit, axis_limit];
outer_vertices_y = [axis_limit, axis_limit, -axis_limit, -axis_limit, axis_limit];
inner_hole_x = R0 * cos(fliplr(theta));
inner_hole_y = R0 * sin(fliplr(theta));
patch([outer_vertices_x NaN inner_hole_x], [outer_vertices_y NaN inner_hole_y], 'w', 'EdgeColor', 'none');

% --- 5. 美化与关闭坐标轴 ---
axis off;
axis tight;

% --- 6. 定制化的颜色条 ---
cb = colorbar('Location', 'southoutside');
fixed_color_range = [1.2, 2.6];
fixed_ticks = [1.2, 1.9, 2.6];
caxis(ax, fixed_color_range);
set(cb, 'Ticks', fixed_ticks, 'FontName', 'Times New Roman', 'FontSize', 11);
cb.Label.String = 'Radius (\mum)';
cb.Label.FontSize = 12;
cb.Label.FontName = 'Times New Roman';

hold(ax, 'off');

end