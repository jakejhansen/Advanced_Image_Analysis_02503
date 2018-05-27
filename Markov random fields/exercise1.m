clear all;
im = double(imread('data/noisy_circles.png'));

seg_G = zeros(size(im));
seg_G(im <= 85) = 1;
seg_G((im > 85) & (im <= 2*85)) = 2;
seg_G(im > 2*85) = 3;


truth = double(imread('data/noise_free_circles.png'));
seg_truth = zeros(size(truth));
seg_truth(truth <= 85) = 1;
seg_truth((truth > 85) & (truth <= 2*85)) = 2;
seg_truth(truth > 2*85) = 3;
imagesc(seg_truth)





seg = seg_G;
imagesc(seg)
%% With alpha expansion 
mu = [70, 130, 190];
U = IMC(im, seg, mu, 0.0005, 1);
U = reshape(U,size(U,1)*size(U,2),3);
[S,iter] = multilabel_MRF(U,size(im), 4, 100);
imagesc(S)

%%

%Plot
figure(1)
subplot(1,3,1);
imagesc(im);
subplot(1,3,2);
imagesc(seg);
subplot(1,3,3);
plot_histograms(im, seg, 3);

alpha = 0.0005; %alpha parameter
beta = 1; %Punish parameter

mu = [70, 130, 190]; %means (as in document)

[likelihood, prior, posterior] = compute_seg_E(im, seg, mu, alpha, beta)