function [PiPj_f, F, Fij] = calculate_force(PiPj, Diffij, Distij, RiRj, ndm)
%calculate_force calculates repulsive forces between interacting particles.
% This version is optimized for modern MATLAB using implicit broadcasting.

ki = 1.0;

% 【核心公式】：计算所有输入对的潜在排斥力大小
F = ki .* max(1.0 - Distij ./ RiRj, 0);

% --- 1. 快速筛选出实际发生接触(力不为零)的粒子对 ---
itouch = F ~= 0;
if ~any(itouch) % 如果没有任何粒子接触，直接返回空值，效率最高
    PiPj_f = [];
    F = [];
    Fij = [];
    return;
end

PiPj_f = PiPj(itouch, :);
F = F(itouch); % F 现在只包含非零的力大小

% --- 2. 仅为接触的粒子对计算力矢量 ---

% 获取接触对的距离大小和距离矢量
Distij_touch = Distij(itouch);
Diffij_touch = Diffij(itouch, :);

% 【数值稳定性处理】：用一个极小的数epsilon替换零距离，防止除零错误
epsilon = 1e-12;
Distij_touch(Distij_touch == 0) = epsilon;

% 计算力的单位方向向量
UnitDirection = Diffij_touch ./ Distij_touch; % 利用隐式广播，(M x ndm) ./ (M x 1)

% 【优化核心】：利用隐式广播计算最终力矢量
% F(:)确保其为列向量(M x 1)，直接与(M x ndm)的UnitDirection相乘
Fij = F(:) .* UnitDirection;

end
