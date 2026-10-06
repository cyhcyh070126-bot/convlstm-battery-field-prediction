function Draw_particle_1(Ps,N,k,L0,ndm,BOX)

if ndm == 2
    %Draw_particle_2D(Ps,N,k,L0);
    Draw_particle_2D_1(Ps,N,k,BOX)
elseif ndm == 3
    Draw_particle_3D_1(Ps,N,k,BOX);
end
