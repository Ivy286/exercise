clc;
clear all;
x = -10:0.02:10;
t1 = -5:0.1:5;
t2 = -0.5:0.01:0.5;
y_sigmoid = 1./(1+exp(-x));
y_deltsigmoid = (exp(-x))./((1+exp(-x)).^2);
%y_tanh = (exp(x)-exp(-x))./(exp(x)+exp(-x));
%y_delttanh = 1-(y_tanh).^2;
plot(x,y_deltsigmoid,'r', 'linewidth',2); title('delt sigmoid');
hold on;
plot(x,y_sigmoid,'b', 'linewidth',2);
hold on;
legend('delt sigmoid','sigmoid');
plot(t1,0,'-','linewidth',2);
%plot(0,t2, '-','linewidth',2);
grid on;
axis on;
ylim([-0.1,1.2]);