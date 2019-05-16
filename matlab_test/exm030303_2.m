function y=exm030303_2(x)
% exm030303_2
L1=x<=1
L2=-1<x&x<=1
L3=1<x
y=zeros(size(x))
y(L1)=x(L1)
y(L2)=x(L2).^3.*cos(2*pi*x(L2))
y(L3)=exp(-x(L3)+1)