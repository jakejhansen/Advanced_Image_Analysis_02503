%Iterative optimization

im = double(imread('data/noisy_circles.png'));

seg_G = zeros(size(im));
seg_G(im <= 100) = 1;
seg_G((im > 100) & (im <= 160)) = 2;
seg_G(im > 160) = 3;


truth = double(imread('data/noise_free_circles.png'));
seg_truth = zeros(size(truth));
seg_truth(truth <= 85) = 1;
seg_truth((truth > 85) & (truth <= 2*85)) = 2;
seg_truth(truth > 2*85) = 3;

mu = [70, 130, 190];
alpha = 0.0005;
beta = 1;

%% Update everything in parallel
seg = seg_G;
num_iterations = 10;


%Do some iterations of the optimization
figure(1);
subplot(1,2,1)
imagesc(seg)
title('Original Segmentation')
for i = 1:num_iterations
    V = IMC(im, seg, mu, alpha, beta);
    %Compute new segmentation
    [mini, seg] = min(V,[],3);
    %figure(2)
    %imagesc(seg)
    %pause;
end
figure(1)
subplot(1,2,2)
imagesc(seg)
title(['Segmentation after ', num2str(num_iterations) , ' iterations'])

%% Update everything 
seg = seg_G;
num_iterations = 10;
checkerboard_mask = checkerboard(1,size(im,1),size(im,2)) > 0.5
checkerboard_mask = checkerboard_mask(1:size(im,1), 1:size(im,2))

figure(1);
subplot(1,2,1)
imagesc(seg)
title('Original Segmentation')
for i = 1:num_iterations
    V = IMC(im, seg, mu, alpha, beta);
    %Compute new segmentation
    [mini, seg_temp] = min(V,[],3);
    seg(checkerboard_mask == 0) = seg_temp(checkerboard_mask == 0);
    checkerboard_mask = ~checkerboard_mask;
    
    %figure(2)
    %imagesc(seg)
    %pause;

 
end
figure(1)
subplot(1,2,2)
imagesc(seg)
title(['Segmentation after ', num2str(num_iterations) , ' iterations'])