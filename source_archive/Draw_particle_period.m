function Draw_particle_period(Ps,N,k,L0,ndm)

if ndm == 2
    move = [repmat([L0,0,0],N,1); ...
            repmat([0,L0,0],N,1); ...
            repmat([L0,L0,0],N,1); ...
            repmat([L0,-L0,0],N,1)];
    move = [move;-move];
    Ps_p = [Ps;repmat(Ps,8,1)+move];
    Draw_particle_2D(Ps_p,N*9,k,L0);
    
elseif ndm == 3
    move = [repmat([L0,0,0,0],N,1); ...
            repmat([0,L0,0,0],N,1); ...
            repmat([L0,L0,0,0],N,1); ...
            repmat([L0,-L0,0,0],N,1)];
    move = [move;-move];
    Ps_p = [Ps;repmat(Ps,8,1)+move];
    move_z = repmat([0,0,L0,0],N*9,1);
    Ps_p = [Ps_p;Ps_p+move_z;Ps_p-move_z];
    Draw_particle_3D(Ps_p,N*27,k,L0);
end
