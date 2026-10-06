function [PiPj_1] = Contact_Detection_By_Boundary_Box1(Ps, ~, ~)
% --- Contact_Detection_By_Boundary_Box1.m (全新升级版) ---
%
% 这是一个高性能、高精度的接触检测函数，用于替换旧的、存在缺陷的同名函数。
% 它结合了两种方法的优点：
% 1. 快速筛选 (AABB): 使用向量化的"轴对齐边界盒"方法，快速剔除大量不可能接触的粒子对。
% 2. 精确检测: 只对筛选后的小部分"嫌疑"粒子对，进行精确的距离计算，保证结果100%正确。
%
% 注意：为了保持与旧版本的兼容性，函数签名保留了(Ps, ndm, N)，
% 但内部实现已不再需要 ndm 和 N，因此用'~'忽略它们。

% --- 0. 准备数据 ---
if isempty(Ps) || size(Ps, 1) < 2
    PiPj_1 = [];
    return;
end
ndm = size(Ps, 2) - 1;
N = size(Ps, 1);
radii = Ps(:, ndm+1);
coords = Ps(:, 1:ndm);

% --- 1. 生成所有可能的粒子对 ---
% nchoosek 自动处理 i < j 的情况，避免重复和自己与自己比较
PiPj_potential = nchoosek(1:N, 2);
p_indices = PiPj_potential(:, 1);
j_indices = PiPj_potential(:, 2);

% --- 2. 快速筛选 (向量化的AABB边界盒检测) ---
% A. 获取所有粒子的边界盒 [min_x, min_y, max_x, max_y]
min_coords = coords - radii;
max_coords = coords + radii;

% B. 提取所有粒子对的边界盒信息
p_min = min_coords(p_indices, :);
p_max = max_coords(p_indices, :);
j_min = min_coords(j_indices, :);
j_max = max_coords(j_indices, :);

% C. 向量化判断边界盒是否重叠
% 两个盒子重叠的条件是：它们在【所有】坐标轴上的投影都重叠。
overlap_x = (p_min(:, 1) <= j_max(:, 1)) & (j_min(:, 1) <= p_max(:, 1));
overlap_y = (p_min(:, 2) <= j_max(:, 2)) & (j_min(:, 2) <= p_max(:, 2));

% D. 找出在所有维度上边界盒都重叠的"嫌疑"粒子对
suspicious_indices = overlap_x & overlap_y;
PiPj_suspicious = PiPj_potential(suspicious_indices, :);

% 如果经过筛选后没有任何嫌疑对，直接返回空集
if isempty(PiPj_suspicious)
    PiPj_1 = [];
    return;
end

% --- 3. 精确检测 (只对"嫌疑"粒子对进行) ---
% 这是旧函数所缺失的、最关键的一步
p_indices_s = PiPj_suspicious(:, 1);
j_indices_s = PiPj_suspicious(:, 2);

% A. 计算嫌疑对的半径和的平方
radii_sum = radii(p_indices_s) + radii(j_indices_s);
radii_sum_sq = radii_sum.^2;

% B. 计算嫌疑对的中心距的平方 (避免开方，速度更快)
diff_coords = coords(p_indices_s, :) - coords(j_indices_s, :);
dist_sq = sum(diff_coords.^2, 2);

% C. 找出中心距平方 < 半径和平方的，这才是【真正接触】的粒子对
actual_contact_mask = dist_sq < radii_sum_sq;
PiPj_1 = PiPj_suspicious(actual_contact_mask, :);

end




