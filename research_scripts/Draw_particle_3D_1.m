function Draw_particle_3D_1(Ps,n,k,BOX)

alpha = 0.8;
colors = hsv(n); % other choices: turbo, hot, cool, jet, autumn, summer, copper, pink
t=linspace(0,2*pi,100);
p=linspace(0,2*pi,100);

figure(k);
hold on; 

%vertices=[0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1]*L0;
 vertices=[0 0 0;BOX(2,1) 0 0;BOX(2,1) BOX(2,2) 0;0 BOX(2,2) 0;0 0 BOX(2,3);BOX(2,1) 0 BOX(2,3);BOX(2,1) BOX(2,2) BOX(2,3);0 BOX(2,2) BOX(2,3)];
%   ;BOX(2,1) BOX(2,2) BOX(2,3)

faces=[1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8;1 2 3 4;5 6 7 8 ];
for i = 1 : 6
    h = patch(vertices(faces(i,:),1),vertices(faces(i,:),2),vertices(faces(i,:),3),'k');
    set(h,'facealpha',0.05);
end

for ip=1:n%(n*7)    
   view([0,0]);
   %xlim([-4 7]);ylim([-4 7]);zlim([-4 7]);  r
   [theta,phi]=meshgrid(t,p);   
   xp=Ps(ip,1)+Ps(ip,4)*sin(theta).*cos(phi); %x轴坐标                  
   yp=Ps(ip,2)+Ps(ip,4)*sin(theta).*sin(phi); %y轴坐标
   zp=Ps(ip,3)+Ps(ip,4)*cos(theta);   

   surf(xp,yp,zp,'facecolor',colors(ip,:),'MeshStyle','none', 'facealpha', alpha); 
   daspect([1,1,1]);                                        %设置xyz轴比例为1:1:1
   %camlight('headlight');                                  %设置默认光照
   %shading interp;
   hold on
   %axis off; 
 end 


% for i = 1 : n
%       x = Ps(i,1) * cos(t) + Ps(i,2);
%       y = Ps(i,1) * sin(t) + Ps(i,3);
%       patch(x, y, colors(i,:), 'facealpha', alpha, 'edgecolor', 'none');
% end
axis equal;
