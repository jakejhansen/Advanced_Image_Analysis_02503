%Exercise 3.1.5
clear all;
im = double(imread('data/V12_10X_x502.png'));
im = im / (2^16 - 1);

colormap gray
histogram(im) %0.4 and 0.7

addpath GraphCut

d = [179 174 182 162 175 165]; % heights (data)
mu = [0.4, 0.7]; % means of two classes
beta = 0.1; % weight of the prior term
w_s = (im(:)-mu(1)).^2; % source weightt
w_t = (im(:)-mu(2)).^2; % sink weights
N = numel(im); % number of graph nodes
indices = (1:N)'; % an index for each person

% terminal and internal edge matrix
E_terminal = [indices,[w_s,w_t]]; 

% Internal matrix construction
v = imageGraph([size(im,1) size(im,2)],4);
test = v.Edges.EndNodes;
E_internal = [test, beta * ones(size(test,1),2)];

[Scut,flow] = GraphCutMex(N,E_terminal,E_internal); % here it happens

% Construct segmentation
seg = ones(size(im));

seg(Scut) = 2;

subplot(1,3,1)
imagesc(im)
title('Original Image')
subplot(1,3,2)
imagesc(seg)
title('Segmentation')
subplot(1,3,3)
plot_histograms(im*255, seg, 3);
title('Histograms')

%%
mu = [0.43, 0.6, 0.65]; % means of two classes
seg_G = zeros(size(im));
seg_G(im <= mu(1)) = 1;
seg_G((im > mu(1)) & (im <= mu(2))) = 2;
seg_G(im > mu(2)) = 3;

seg = seg_G;
U = IMC(im, seg, mu, 0.0005, 1);
U = reshape(U,N,3);
[S,iter] = multilabel_MRF(U,size(im), 4, 100);
colormap gray;
subplot(1,3,1)
imagesc(im)
title('Original Image')
subplot(1,3,2)
imagesc(seg_G)
title('Thresholded Segmentation')
subplot(1,3,3)
imagesc(S)
title('Alpha expandend segmentation')

plot_histograms(im*255, S, 3);
