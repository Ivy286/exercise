clf;t=6*pi*(0:100)/100;y=1-exp(-0.3*t).*cos(0.7*t);
tt=t(find(abs(y-1)>0.05)); %寻找大于0.05的元素
ts=max(tt); ％寻找tt中最大的元素
plot(t,y,'r-','LineWidth',3)
axis([-inf,6*pi,0.6,inf])
set(gca,'Xtick',[2*pi,4*pi,6*pi],'Ytick',[0.95,1,1.05,max(y)])
grid on
title('\it y = 1 - e^{ -\alphat}cos{\omegat}')
text(13.5,1.2,'\fontsize{12}{\alpha}=0.3')
text(13.5,1.1,'\fontsize{12}{\omega}=0.7')
hold on;plot(ts,0.95,'bo','MarkerSize',10);hold off
cell_string{1}='\fontsize{12}\uparrow';
cell_string{2}='\fontsize{16} \fontname{隶书}镇定时间';
cell_string{3}='\fontsize{6} ';
cell_string{4}=['\fontsize{14}\rmt_{s} = ' num2str(ts)];
text(ts,0.85,cell_string)
xlabel('\fontsize{14} \bft \rightarrow')
ylabel('\fontsize{14} \bfy \rightarrow')