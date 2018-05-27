function [] = plot_histograms(im, seg_G, BW)
colormap gray
figure(1)
H_tot = histogram(im, 'FaceColor', [0,0,0], 'BinWidth', BW);
xlim([0, 256]);

handles.H = figure(2);
for i = 1:3
    figure(2);
    H = histogram(im(seg_G == i),'BinWidth',BW);
    
    figure(1)
    hold on;
    x = H.BinWidth/2 + H.BinLimits(1) : H.BinWidth : H.BinLimits(2)-H.BinWidth/2;
    y = H.BinCounts;
    plot([x(1),x, x(end)],[0, y, 0], 'LineWidth', 4);
    
end
close(handles.H);
end

