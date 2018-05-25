%% Exercise 2.5 and 2.6
%clear all
image = 'CT_lab_high_res.png';

if strcmp(image, 'SEM.png')
    im = double(imread(image));
    im2 = im;
    im = im(1:690, :);
    im(im < 100) = 0;
    im(im >= 100) = 255;
    t = [30];
    thresh = 0.1;
end


if strcmp(image, 'CT_lab_high_res.png')
    im = double(imread(image));
    im = double(uint8(255*mat2gray(im)));
    im2 = im;
    im(im < 150) = 0;
    im(im >= 150) = 255;
    t = [30];
    thresh = 0.2;
end

if strcmp(image, 'CT_lab_low_res.png')
    im = double(imread(image));
    im = double(uint8(255*mat2gray(im)));
    im2 = im;
    im(im < 170) = 0;
    im(im >= 170) = 255;
    t = [5];
    thresh = 0.1;
end

if strcmp(image, 'CT_lab_med_res.png')
    im = double(imread(image));
    im = double(uint8(255*mat2gray(im)));
    im2 = im;
    im(im < 140) = 0;
    im(im >= 140) = 255;
    t = [10];
    thresh = 0.1;
end

if strcmp(image, 'CT_synchrotron.png')
    im = double(imread(image));
    im = double(uint8(255*mat2gray(im)));
    im2 = im;
    im(im < 130) = 0;
    im(im >= 130) = 255;
    t = [18];
    thresh = 0.05;
end

if strcmp(image, 'Optical.png')
    im = double(imread(image));
    im = double(uint8(255*mat2gray(im)));
    im(1:50, 1:170) = 0; 
    im2 = im;
    im(im < 200) = 0;
    im(im >= 200) = 255;
    t = [25];
    thresh = 0.05;
end

for i = 1:length(t)
    [filt, filt_dd] = compute_g(t(i));
    Lxx = filter2(filt_dd, im);
    Lxx = filter2(filt', Lxx);
    
    Lyy = filter2(filt_dd', im);
    Lyy = filter2(filt, Lyy);
    
    L(:,:,i) = t(i)*(Lxx + Lyy);

    maxi = imregionalmax(L(:,:,i));
    mini = imregionalmin(L(:,:,i));
    
    maxi = maxi & (L > thresh*max(max(L(:,:,i))));
    mini = mini & (L < thresh*min(min(L(:,:,i))));
    
    
    [a b] = find(mini(:,:,i));
    cx{i} = a;
    cy{i} = b;
end

totblobs = sum(cellfun('length',cx))


set(0,'defaultAxesFontSize',20)

figure(1)

colormap gray
a1 = subplot(2,2,1);
imagesc(im2)
title(['Original image: ', image])
a2 = subplot(2,2,2);
imagesc(im);
title('Original image (thresholded)')
a3 = subplot(2,2,3);
imagesc(L)
title('Filtered image (L)')
a4 = subplot(2,2,4);
imagesc(im2)
hold on;
for i = 1:length(t)
    viscircles([cy{i}, cx{i}], ceil(ones(length(cx{i}),1) * sqrt(2*t(i))));
end
title({['T = ', num2str(t)], [' Number of blobs:', num2str(totblobs)]});
linkaxes([a1,a2,a3,a4],'xy')