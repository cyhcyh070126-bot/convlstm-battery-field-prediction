function Draw_particle_Cricle_3D(Ps,n,k,R)

alpha = 0.8;
colors = hsv(n); % other choices: turbo, hot, cool, jet, autumn, summer, copper, pink
t=linspace(0,2*pi,100);
p=linspace(0,2*pi,100);

figure(k);
hold on; 

h = 0; % 高度
r = R;  %半径
pos = [0,0]; % 圆心位置
tt=0:0.001:(2*pi);  % 圆滑性设置
tt=[tt,0];
plot3(pos(1)+r*sin(tt),pos(2)+r*cos(tt), h*ones(size(tt)))




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



axis equal;