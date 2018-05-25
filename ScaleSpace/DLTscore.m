% This function evaluates as a score function the accuracy of H
% input:
%    @param x1 = x; the data values; organised either as N*[2 or 3] or [2 or 3]*N matrix
%    @param x2 = x'; the reference values; organised either as N*[2 or 3] or [2 or 3]*N matrix
%    @param H = homogaphy matrix
%    @param t = tolerance / threshold / ratio; some parameter to distinguish in- and outliers
% output:
%    @argout inliers = point indices for pairs within a margin of error after d(x',Hx)
%    @argout outliers = point indices for pairs outside the margin of error after d(x',Hx)
%    @argout sumDistance = the sum of squared distances of d(x', Hx)
function [inliers, outliers, sumDistance] = DLTscore(x1, x2, H, t)
    % iterate over number of points in [x1, x2]

    % compute x* = Hx; 
    % normalize the results for 2D projection ( [x y z] => [u v 1.0])

    % compute d(x',x*)

    % classify pairs in inlier and outlier
    % with respect to some tolerance parameter t
    % (you can define what that shall be)

