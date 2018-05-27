%% 1: Read in movie data
clear all;
vid = VideoReader('data/echiniscus.mp4')

for i = 1:(vid.Duration*vid.FrameRate)
    f = im2frame(vid.readFrame);
    frame(:,:,:,i) = f.cdata;
end

%% 2: Process frames
im_gray = [];
for i = 1:size(frame,4)
    im = frame(:,:,:,i);
    im = double(im);
    im = im/255;
    r = im(:,:,1);
    g = im(:,:,2);
    b = im(:,:,3);
    im_gray(:,:,i) = (2*b-(r+g)+2)/4;
end

%% Check that it works
%for i = 1:size(im_gray,3)
%   imagesc(im_gray(:,:,i));
%  pause;
%end

%% 3: Initialize Snake -- echniscus
im = im_gray(:,:,60);
imagesc(im);
hold on;
[x,y] = compute_circle(170,140,60, 100);
update_size = 90;
plot(x, y, 'r');

alpha = 0.05;
beta = 2;
disp = false;

for i = 1:40
    clf()
    [x,y] = get_snake(x,y, im, im_gray, update_size, disp, alpha, beta);
    drawnow;
end
%% Read in next image a recompute
update_size = 15;
    
for index = 200:size(im_gray,3)
    clf()
    im = im_gray(:,:,index);
    for i = 1:20
       [x,y] = get_snake(x,y, im, im_gray, update_size, disp, alpha, beta);
    end
    drawnow;
end
