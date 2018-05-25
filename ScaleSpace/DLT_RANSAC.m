% This function performs a direct linear transform (DLT) in the framework
% of random sampling and consensus (RANSAC) from point pairs in 2D, such that x'=H*x
% input:
%    @param x1 = x; the data values; organised either as N*[2 or 3] or [2 or 3]*N matrix
%    @param x2 = x'; the reference values; organised either as N*[2 or 3] or [2 or 3]*N matrix
% output:
%    @argout h = column vector of free parameters of DLT
%    @argout H = homogaphy matrix, composed of elemets of "h"
%    @argout inliers = point indices for pairs within a margin of error after d(x',Hx)
%    @argout outliers = point indices for pairs outside the margin of error after d(x',Hx)
function [H, h, inliers, outliers] = DLT_RANSAC(x1, x2)
    % set "M", the number of candidate points used per random sample
    % (e.g. the size of the sample).

    % set "N", the number of random sample iterations until reaching
    % consensus. This parameter is flexible ...

    for n = 1:N
        % generate a random sample of your input points.
        % remember to avoid double inclusions.

        % run DLT for single sample
        [H, h] = DLT(); % to-be-filled
    
        % evaluate the sample and the accuracy of H.
        % if the sample is better than previous runs,
        % save its results. Store inlier (here or after the iterations
        % in the end.
    end
	
