function Draw_particle_Cricle(Ps, N, R0, ndm)
%DRAW_PARTICLE_CRICLE "调度员"函数 (修正版)
%   根据维度(ndm)调用正确的绘图函数。

if ndm == 2
    % 用简化的参数调用我们修正后的2D绘图函数
    Draw_particle_Cricle_2D(Ps, N, R0);

elseif ndm == 3
    % 这里为你未来的3D绘图函数留出接口
    % Draw_particle_Cricle_3D(Ps, N, R0);
    disp('3D plotting function is not yet implemented.');

end

end
