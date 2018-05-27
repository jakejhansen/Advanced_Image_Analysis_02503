function X_new = extended_kernel(X, lambda, alpha, beta)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
L = zeros(size(X,1));
alpha = alpha; %lower values reduces shrinkage
beta = beta; %Makes more smooth for higher values
kern = alpha * [0 1 -2 1 0] + beta * [-1 4 -6 4 -1];

L(1,1:3) = kern(3:5);
L(1,end-1:end) = kern(1:2);
L(2,1:4) = kern(2:5);
L(2,end) = kern(1);
L(end-1, 1) = kern(5);
L(end-1, end-3:end) = kern(1:4);
L(end, 1:2) = kern(4:5);
L(end, end-2:end) = kern(1:3);

for i = 3:size(L,1)-2
    L(i,i-2:i+2) = kern;
end

%Find X_news
lambd = lambda;
X_new = inv(eye(size(L)) - lambd * L) * X;

end

