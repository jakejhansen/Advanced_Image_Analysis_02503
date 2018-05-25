% This function performs a direct linear transform (DLT) for
% point pairs in 2D, such that x'=H*x
% input:
%    @param x1 = x; the data values; organised either as N*[2 or 3] or [2 or 3]*N matrix
%    @param x2 = x'; the reference values; organised either as N*[2 or 3] or [2 or 3]*N matrix
% output:
%    @argout h = column vector of free parameters of DLT
%    @argout H = homogaphy matrix, composed of elemets of "h"
function [H, h] = DLT(x1, x2)

    % check row- and column order of x1 and x2
    % suggestion: for each point being 1 column vector
    % allows for an easier matrix composition of A.
    % Also check that you have enough dimensions for your
    % formulation of A.
    
    % check that you have enough points to do a linear
    % least squares (i.e. prevent under-determined system).

    
    % setup your equation system in matrix form (i.e. construct A)
    A=zeros();


    [U,D,V] = svd(A);

    % compute vector h and homography matrix H



