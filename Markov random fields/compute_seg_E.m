function [likelihood, prior, posterior] = compute_seg_E(im, seg, mu, alpha, beta)

seg_flat = seg(:); %flattened segmentation for indexing
im_flat = im(:); %flattened image for indexing

V1 = 0;
for i = 1:size(im_flat,1)
   f = seg_flat(i);
   V1 = V1 + alpha * (mu(f) - im_flat(i))^2;
end


V2 = 0;
V2 = V2 + beta*sum(sum(seg(:,1:end-1)-seg(:,2:end) ~= 0)); %Compare in x
V2 = V2 + beta*sum(sum(seg(1:end-1,:)-seg(2:end,:) ~= 0)); %Compare in y

likelihood = V1;
prior = V2;
posterior = V1+V2;


end

