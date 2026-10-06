function Draw_particle_2D_1(Ps,n,k,BOX)

alpha = 0.8;
colors = hsv(n); % other choices: turbo, hot, cool, jet, autumn, summer, copper, pink
t = 0 : 0.1 : 2*pi;

figure(k);
hold on; 

rectangle('Position', [BOX(1,:),BOX(2,:)-BOX(1,:)]);


for i = 1 : n
      x = Ps(i,3) * cos(t) + Ps(i,1);
      y = Ps(i,3) * sin(t) + Ps(i,2);
      patch(x, y, colors(i,:), 'facealpha', alpha, 'edgecolor', 'none');
end
axis equal;
