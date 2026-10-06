function [Ps]=Boundary_Particle_back(Ps,BOX,ndm)
%把超出边界的粒子移动到相反的位置，并设置虚拟粒子
%返回值为：粒子当前位置坐标及虚拟粒子当前位置坐标

ifout_low = Ps(:,1:ndm) < BOX(1,:);
while any(ifout_low,'all')
    Ps(:,1:ndm) = Ps(:,1:ndm) + (BOX(2,:)-BOX(1,:)).*ifout_low;
    ifout_low = Ps(:,1:ndm) < BOX(1,:);
end

ifout_high = Ps(:,1:ndm) > BOX(2,:);
while any(ifout_high,'all')
    Ps(:,1:ndm) = Ps(:,1:ndm) - (BOX(2,:)-BOX(1,:)).*ifout_high;
    ifout_high = Ps(:,1:ndm) > BOX(2,:);
end