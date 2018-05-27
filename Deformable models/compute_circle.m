function [x, y] = circle(x,y,r, n)
th = 0:2*pi/n:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
%h = plot(xunit, yunit, 'r*');
x = xunit;
y = yunit;
end