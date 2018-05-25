%% Exercise 2.2

[filt, filt_dd] = compute_g(5);

im = imread('test_blob_uniform.png');

imOut = filter2(filt,im);

imOut = filter2(filt', imOut);

figure()
subplot(1,2,1);
imagesc(im)
subplot(1,2,2);
imagesc(imOut)


%% Exercise 2.3
close all
im = double(imread('test_blob_uniform.png'));
t = 100

for t = 1:10:1200
%Get filter and apply it to compute the laplacian
[filt, filt_dd] = compute_g(t);
Lxx = filter2(filt_dd, im);
Lxx = t*filter2(filt', Lxx);

Lyy = filter2(filt_dd', im);
Lyy = t*filter2(filt, Lyy);

L = Lxx + Lyy;

maxi = imregionalmax(L);
mini = imregionalmin(L);

maxi = maxi & (L > 0.9*max(max(L)));
mini = mini & (L < 0.9*min(min(L)));

[cx, cy] = find(mini | maxi);

set(0,'defaultAxesFontSize',20)

figure(1)
colormap gray
subplot(2,2,1);
imagesc(im)
title('Original Image')
subplot(2,2,2);
imagesc(L);
title('Smoothed image (L)')
subplot(2,2,3);
imagesc(mini | maxi)
title('Min and maxima of smoothed image')
subplot(2,2,4);
imagesc(im)
hold on;
viscircles([cy, cx], ceil(ones(length(cx),1) * sqrt(2*t)));
title(['T = ', num2str(t)]);
pause
end
