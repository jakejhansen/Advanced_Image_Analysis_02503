% Function for plotting sift descriptors
% 
% function plotSift(im, ip)
% 
% Input
%   im - image
%   ip - interest points with 4 x n, where first row is column, second row
%       coordinate is row coordinate, and third row is scale
% 

function plotSift(im, ip)

nPt = 100;
x = 1:nPt;
c = 2*pi*x/(nPt-1);
cPt = [cos(c); sin(c)];

%figure('Color',[1, 1, 1])

imagesc(im);
colormap gray;
hold on
axis image
for i = 1:size(ip,2)
    plot(ip(1,i) + cPt(1,:)*ip(3,i), ip(2,i) + cPt(2,:)*ip(3,i), 'g', 'LineWidth', 1.2);
end








