function Ps_Virc=Circle_Boundary(R0,ndm)
% clc
% clear all
% R0=11;
% ndm=3;

kn=floor(R0*pi/2);
if ~isfinite(R0) || kn < 1
    error('Boundary radius is too small for the original boundary discretization.');
end
if ndm==3
    phi=R0;
    alpha=pi*(0:kn)/kn;
%     theta=2*pi*(1:2*kn)/kn;

    k=1;
    Ps_Virc(k,:)=[0,0,phi,1];
    k=k+1;
    for i=1:kn
        kn2=round(2*kn*sin(alpha(i)));
        theta=2*pi*(1:kn2)/kn2;
        for j=1:kn2
            Ps_Virc(k,1)=phi*sin(theta(j)).*sin(alpha(i));
            Ps_Virc(k,2)=phi*cos(theta(j))*sin(alpha(i));
            Ps_Virc(k,3)=phi*cos(alpha(i));
            Ps_Virc(k,4)=1;
            k=k+1;
        end
    end
    Ps_Virc(k,:)=[0,0,-phi,1];
end
% Draw_particle_Cricle(Ps_Virc,size(Ps_Virc,1),1,R0,ndm);
if ndm==2
    phi=R0;
    theta=2*pi*(0:(2*kn))/(2*kn);
    Ps_Virc(:,1)=phi.*sin(theta);
    Ps_Virc(:,2)=phi.*cos(theta);
    Ps_Virc(:,3)=1;
end
% Draw_particle_Cricle(Ps_Virc,size(Ps_Virc,1),1,R0,ndm);
