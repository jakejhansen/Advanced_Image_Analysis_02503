%% 1: Read in movie data
clear all;
vid = VideoReader('data/crawling_amoeba.mov')

for i = 1:(vid.Duration*vid.FrameRate)
    f = im2frame(vid.readFrame);
    frame(:,:,:,i) = f.cdata;
end

%% 2: Process frames
im_gray = [];
for i = 1:size(frame,4)
    im = frame(:,:,:,i);
    im = double(rgb2gray(im));
    im = im/max(im(:));
    im_gray(:,:,i) = im;
end

%% Check that it works
%for i = 1:size(im_gray,3)
%   imagesc(im_gray(:,:,i));
%  pause;
%end

%% 3: Initialize Snake
im = im_gray(:,:,1);
imagesc(im);
hold on;
[x,y] = compute_circle(170,150,70, 100);
update_size = 30;
plot(x, y, 'r');

alpha = 0.01;
beta = 0.2;
disp = false;
for i = 1:10
    [x,y] = get_snake(x,y, im, im_gray, update_size, disp, alpha, beta);
    drawnow;
end
%% Read in next image a recompute
update_size = 15;

for index = 2:size(im_gray,3)
    clf()
    im = im_gray(:,:,index);
    for i = 1:5 
       [x,y] = get_snake(x,y, im, im_gray, update_size, disp, alpha, beta);
    end
    drawnow;
end