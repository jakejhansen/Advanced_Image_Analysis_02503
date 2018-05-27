clear all;
close all;

%% Load Image
FileTif='data/nerves_part.tiff';
InfoImage=imfinfo(FileTif);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);
FinalImage=zeros(nImage,mImage,NumberImages,'uint16');

TifLink = Tiff(FileTif, 'r');
for i=1:NumberImages
    TifLink.setDirectory(i);
    FinalImage(:,:,i)=TifLink.read();
end
TifLink.close();

imF = double(FinalImage);
imF = imF/255;
%% Display first image
imagesc(imF(:,:,1))

im_gray = imF;

im = im_gray(:,:,1);
imagesc(im);
hold on;
[x,y] = compute_circle(95,225, 26, 50);
[x2,y2] = compute_circle(230,280, 20, 50);
[x3,y3] = compute_circle(285, 262, 23, 50);
update_size = 10;
plot(x, y, 'b');
plot(x2, y2, 'r');
plot(x3, y3, 'black');

alpha = 0.5;
beta = 1;
disp = false;

%%
for i = 1:15
    hold on;
    imagesc(im)
    [x,y] = get_snake_nerve(x,y, im, im_gray, update_size, disp, alpha, beta, 'b');
    [x2,y2] = get_snake_nerve(x2,y2, im, im_gray, update_size, disp, alpha, beta, 'r');
    [x3,y3] = get_snake_nerve(x3,y3, im, im_gray, update_size, disp, alpha, beta, 'black');
    drawnow;
end

%% Continue with more images
curve = [];
curve2 = [];
curve3 = [];
for index = 1:size(im_gray, 3)
    im = im_gray(:,:,index);
    clf()
    imagesc(im)
    hold on;
    for i = 1:5
        [x,y] = get_snake_nerve(x,y, im, im_gray, update_size, disp, alpha, beta, 'b');
        [x2,y2] = get_snake_nerve(x2,y2, im, im_gray, update_size, disp, alpha, beta, 'r');
        [x3,y3] = get_snake_nerve(x3,y3, im, im_gray, update_size, disp, alpha, beta, 'black');
        %drawnow
    end
    curve(:,:,index) = [x; y];
    curve2(:,:,index) = [x2; y2];
    curve3(:,:,index) = [x3; y3];
    drawnow
end

%%
clf()
pointpairs = [];
imagesc(im_gray(:,:,1));
hold on;
dim = size(curve,3)
Z = 0:dim-2;
Z = Z';
for i = 1:51
    X = curve(1,i,2:dim);
    X = X(:);
    Y = curve(2,i,2:dim);
    Y = Y(:);
    plot3(X,Y,fliplr(Z),'b','linewidth',1 )
    pointpairs = [pointpairs ; X , Y , Z];
end

for i = 1:51
    X = curve2(1,i,2:dim);
    X = X(:);
    Y = curve2(2,i,2:dim);
    Y = Y(:);
    plot3(X,Y,fliplr(Z), 'r','linewidth',1)
    %pointpairs = [pointpairs ; X , Y , Z];
end

for i = 1:51
    X = curve3(1,i,2:dim);
    X = X(:);
    Y = curve3(2,i,2:dim);
    Y = Y(:);
    plot3(X,Y,fliplr(Z), 'black','linewidth',1)
    %pointpairs = [pointpairs ; X , Y , Z];
end

%%
p = pointpairs;
[t]=MyCrustOpen(p);

hold on;
title('Output Triangulation','fontsize',14);
axis equal;
trisurf(t,p(:,1),p(:,2),p(:,3),'facecolor','c','edgecolor','b')