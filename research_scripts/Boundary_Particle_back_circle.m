function [Ps]=Boundary_Particle_back_circle(Ps,R0,ndm)
%出去的移过去
if ndm==2
    ifout_low =sqrt(Ps(:,1).^2+Ps(:,2).^2)> R0+Ps(:,3);
    while any(ifout_low,'all')
        radial_norm = sqrt(Ps(:,1).^2+Ps(:,2).^2);
        Ps(:,1) = Ps(:,1) - 2*R0.*ifout_low.*Ps(:,1)./max(radial_norm, realmin);
        Ps(:,2) = Ps(:,2) - 2*R0.*ifout_low.*Ps(:,2)./max(radial_norm, realmin);
        ifout_low=sqrt(Ps(:,1).^2+Ps(:,2).^2)> R0+Ps(:,3);
    end
end

if ndm==3
    ifout_low =sqrt(Ps(:,1).^2+Ps(:,2).^2+Ps(:,3).^2)> R0+Ps(:,4);
    while any(ifout_low,'all')
        radial_norm = sqrt(Ps(:,1).^2+Ps(:,2).^2+Ps(:,3).^2);
        Ps(:,1) = Ps(:,1) - 2*R0.*ifout_low.*Ps(:,1)./max(radial_norm, realmin);
        Ps(:,2) = Ps(:,2) - 2*R0.*ifout_low.*Ps(:,2)./max(radial_norm, realmin);
        Ps(:,3) = Ps(:,3) - 2*R0.*ifout_low.*Ps(:,3)./max(radial_norm, realmin);
        ifout_low=sqrt(Ps(:,1).^2+Ps(:,2).^2+Ps(:,3).^2)> R0+Ps(:,4);
    end
end