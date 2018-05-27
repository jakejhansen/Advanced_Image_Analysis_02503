function V = IMC(im, seg, mu, alpha, beta)

%Allocation
V = zeros(size(im,1),size(im,2),length(mu));

%Pad image for neighbour lookup
seg_n = zeros(size(seg,1) + 2, size(seg,2) + 2);
seg_n(2:end-1, 2:end-1) = seg;

%Posterior Energy

for i = 2:size(seg_n,1)-1
    for j = 2:size(seg_n,2)-1
        for f = 1:length(mu)
            V_f = alpha * (mu(f) - im(i-1,j-1))^2;
            f_neighbours = [ seg_n(i,j-1), seg_n(i,j+1), seg_n(i-1, j), seg_n(i+1,j)];
            f_neighbours = f_neighbours(f_neighbours ~= 0);
            V_f = V_f + sum(f_neighbours ~= f);
            V(i-1,j-1,f) = V_f;
        end
    end
end
end