%% 2_1 and 2_2
run('C:\Users\Jake\Documents\MATLAB\vlfeat-0.9.20\toolbox\vl_setup.m')

ImL = imread('CT_lab_high_res.png');
ImR = imread('CT_lab_med_res.png');

[fa, da] = vl_sift(single(ImL));
[fb, db] = vl_sift(single(ImR));

[matches, scores] = vl_ubcmatch(da, db);
numMatches=size(matches,2);

dh1 = max(size(ImR,1)-size(ImL,1),0) ;
dh2 = max(size(ImL,1)-size(ImR,1),0) ;


figure(1) ; clf ;
subplot(1,1,1) ;
imagesc([padarray(ImL,dh1,'post') padarray(ImR,dh2,'post')]) ;
o = size(ImL,2) ;
line([fa(1,matches(1,:));fb(1,matches(2,:))+o], ...
     [fa(2,matches(1,:));fb(2,matches(2,:))]) ;
title(sprintf('%d tentative matches', numMatches)) ;
axis image off ;


figure(2)
subplot(1,2,1);
plotSift(ImL, fa);
subplot(1,2,2);
plotSift(ImR, fb);

%% 2_3
K = 10;

sel = randperm(size(da,2));
sel = sel(1:K);
subplot(1,2,1);
imagesc(ImL); colormap gray, axis off, axis image
vl_plotsiftdescriptor(da(:,sel), fa(:,sel));

sel = randperm(size(db,2));
sel = sel(1:K);
subplot(1,2,2);
imagesc(ImR); colormap gray, axis off, axis image
vl_plotsiftdescriptor(db(:,sel), fb(:,sel));

%% 2_4
da = double(da);
db = double(db);
dan = da./sqrt((ones(size(da,1),1)*sum(da.*da)));
dbn = db./sqrt((ones(size(db,1),1)*sum(db.*db)));


%% 2_5 Matching descriptors - Calculate
%matches1to2 = match_1to2(dan, dbn);
matches2to1 = match_1to2(dbn, dan);

%% 2_5 Matching descriptors - Plot
num_matches = size(matches1to2,2);
matches = [];

for i = 1:num_matches
    matches = [matches, [i, matches1to2{i}(1)]'];
end

figure(1) ; clf ;
subplot(1,1,1) ;
imagesc([padarray(ImL,dh1,'post') padarray(ImR,dh2,'post')]) ;
o = size(ImL,2) ;
line([fa(1,matches(1,:));fb(1,matches(2,:))+o], ...
     [fa(2,matches(1,:));fb(2,matches(2,:))]) ;
title(sprintf('All matches \n %d tentative matches', size(matches,2))) ;
axis image off ;

%%
%With similaity measure
num_matches = size(matches1to2,2);
matches = [];
threshold = 0.6;
for i = 1:num_matches
    % Calculate the ratio between largest and smallest match
    condition = norm(dan(:,i) - dbn(:,matches1to2{i}(1))) / norm(dan(:,i) - dbn(:,matches1to2{i}(2)));
    if (condition < threshold)
       matches = [matches, [i, matches1to2{i}(1)]']; 
    end
end

figure(2) ; clf ;
subplot(1,1,1) ;
imagesc([padarray(ImL,dh1,'post') padarray(ImR,dh2,'post')]) ;
o = size(ImL,2) ;
line([fa(1,matches(1,:));fb(1,matches(2,:))+o], ...
     [fa(2,matches(1,:));fb(2,matches(2,:))]) ;
title(sprintf('Tresholded matches \n %d tentative matches \n Threshold: %.1f', size(matches,2), threshold)) ;
axis image off ;

%%
%With symmetry measure
num_matches = size(matches1to2,2);
matches = [];
for i = 1:num_matches
    % Calculate the ratio between largest and smallest match
    best_match12 = matches1to2{i}(1);
    best_match21 = matches2to1{best_match12}(1);
    
    condition = (best_match21 == i)
    if (condition)
       matches = [matches, [i, matches1to2{i}(1)]']; 
    end
end

figure(3) ; clf ;
subplot(1,1,1) ;
imagesc([padarray(ImL,dh1,'post') padarray(ImR,dh2,'post')]) ;
o = size(ImL,2) ;
line([fa(1,matches(1,:));fb(1,matches(2,:))+o], ...
     [fa(2,matches(1,:));fb(2,matches(2,:))]) ;
title(sprintf('Symmetric matches \n %d tentative matches', size(matches,2))) ;
axis image off ;
%% Question3 - Ransac, own code from previous course #OPTIONAL
numMatches=size(matches,2);
q1tot = [fa(1,matches(1,:));fa(2,matches(1,:));ones(1,size(matches,2))];
q2tot = [fb(1,matches(2,:));fb(2,matches(2,:));ones(1,size(matches,2))];

%Minimum number of matches
nb_matches = 4;

best_model = [];
best_model_percent = 0;

percent_inliers = [];

for h = 1:1000
    consensus = 0;
    i = randperm(size(matches,2),nb_matches)';

    q1 = [fa(1,matches(1,i));fa(2,matches(1,i));ones(1,nb_matches)];
    q2 = [fb(1,matches(2,i));fb(2,matches(2,i));ones(1,nb_matches)];

    H = Hest(q1,q2);
    diff_q1_to_q2 = homtransform(H*q1tot)-homtransform(q2tot);
    diff_q2_to_q1 = homtransform(H^(-1)*q2tot)-homtransform(q1tot);
    matchinlier = [];
    for b = 1:numMatches
       dist = norm(diff_q1_to_q2(:,b))+norm(diff_q2_to_q1(:,b));
       if dist < 7.3423 % sqrt(5.99*3^2) Table 5.7 p, 102
          consensus = consensus + 1; 
          matchinlier = [matchinlier, b];
       end
    end
    percent_inliers = [percent_inliers, consensus/numMatches];
    if percent_inliers(h)*100 > best_model_percent
       best_model_percent = percent_inliers(h)*100;
       best_model = H;
       best_matches_index = matchinlier;
    end
end

max_inliers = max(percent_inliers)*100
mean_inliers = mean(percent_inliers)*100
min_inliers = min(percent_inliers)*100

numMatches=size(best_matches_index,2);

dh1 = max(size(ImR,1)-size(ImL,1),0) ;
dh2 = max(size(ImL,1)-size(ImR,1),0) ;

figure(1) ; clf ;
subplot(1,1,1) ;
imagesc([padarray(ImL,dh1,'post') padarray(ImR,dh2,'post')]) ;
o = size(ImL,2) ;
line([fa(1,matches(1,best_matches_index));fb(1,matches(2,best_matches_index))+o], ...
     [fa(2,matches(1,best_matches_index));fb(2,matches(2,best_matches_index))]) ;
title(sprintf('%d tentative matches', numMatches)) ;
axis image off ;

%% Estimate homography, using ubc match
close all
nb_matches=size(best_matches_index,2);
q1 = [fa(1,matches(1,best_matches_index));fa(2,matches(1,best_matches_index));ones(1,nb_matches)];
q2 = [fb(1,matches(2,best_matches_index));fb(2,matches(2,best_matches_index));ones(1,nb_matches)];

H = Hest(q1,q2);

%% Estimate homography, using threshold
close all
nb_matches=size(matches,2);
q1 = [fa(1,matches(1,:));fa(2,matches(1,:));ones(1,nb_matches)];
q2 = [fb(1,matches(2,:));fb(2,matches(2,:));ones(1,nb_matches)];

H = Hest(q1,q2);

%H = homography2d(q1,q2);



%%
clear L
im = double(ImL);
im = double(uint8(255*mat2gray(im)));
im2 = im;
im(im < 150) = 0;
im(im >= 150) = 255;
t = [30];
thresh = 0.2;


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

p1 = [];

for i = 1:size(cx{1},1)
    p1 = [p1, [cx{1}(i) ; cy{1}(i); 1]]; 
end

transformed = H*p1;
p_trans = [];

for i = 1:size(cx{1},1)
    pp = transformed(:,i);
    pp = pp/pp(3);
    p_trans = [p_trans, pp];
end


cx_other{1} = p_trans(1,:)';
cy_other{1} = p_trans(2,:)';

%Compute features for R image
im = double(ImR);
im = double(uint8(255*mat2gray(im)));
im2 = im;
im(im < 140) = 0;
im(im >= 140) = 255;
t = [10];
thresh = 0.1;
clear L;
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


t = [10];
figure(1)
colormap gray
imagesc(ImR)
hold on;


for i = 1:length(t)
    viscircles([cy{i}, cx{i}], ceil(ones(length(cx{i}),1) * sqrt(2*t(i))), 'Color', 'g', 'LineStyle',':', 'LineWidth', 1);
end

for i = 1:length(t)
    viscircles([cy_other{i}, cx_other{i}], ceil(ones(length(cx_other{i}),1) * sqrt(2*t(i))));
end
