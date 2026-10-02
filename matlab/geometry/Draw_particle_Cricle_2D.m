function Draw_particle_Cricle_2D(Ps, N, R0)
%DRAW_PARTICLE_CRICLE_2D 真正的2D粒子可视化函数 (修正版)
%   此函数只负责在当前激活的图窗上绘图，绝不自己创建新图窗。
%   输入参数已被简化为 Ps, N, R0。

% --- 关键修正：删除了 figure(k) 指令 ---

hold on; % 保持当前图形，以便叠加绘制

% 沿用你原来的hsv颜色方案，并为粒子设置透明度
alpha = 0.7; % 70% 的不透明度，让重叠部分可见
colors = hsv(N);

% 绘制所有粒子
t = 0 : 0.1 : 2*pi; % 定义画圆所需的角度点
for i = 1 : N
    % 根据每个粒子的中心 (Ps(i,1), Ps(i,2)) 和半径 Ps(i,3) 来计算圆周坐标
    x = Ps(i,3) * cos(t) + Ps(i,1);
    y = Ps(i,3) * sin(t) + Ps(i,2);
    % 使用 patch 函数绘图，并设置透明度
    patch(x, y, colors(i,:), 'FaceAlpha', alpha, 'EdgeColor', 'none');
end

% 绘制外边界圆
rectangle('Position', [-R0, -R0, 2*R0, 2*R0], 'Curvature', [1 1], 'EdgeColor', 'k', 'LineWidth', 2);

axis equal; % 确保坐标轴比例一致，圆看起来是圆的
axis_limit = R0 * 1.1;
xlim([-axis_limit, axis_limit]);
ylim([-axis_limit, axis_limit]);

hold off; % 结束在当前图窗上的绘制

end
