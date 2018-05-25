function [ H ] = Hest(q1,q2)
%Make points homogene
q1 = q1./q1(3,:);
q2 = q2./q2(3,:);

%Normalize data
Mean1=mean(q1)';
q1(1,:)=q1(1,:)-Mean1(1);
q1(2,:)=q1(2,:)-Mean1(2);
S1=mean(sqrt(diag(q1'*q1)))/sqrt(2);
q1(1:2,:)=q1(1:2,:)/S1;
T1=[eye(2)/S1,-Mean1(1:2)/S1;0 0 1];

Mean2=mean(q2)';
q2(1,:)=q2(1,:)-Mean2(1);
q2(2,:)=q2(2,:)-Mean2(2);
S2=mean(sqrt(diag(q2'*q2)))/sqrt(2);
q2(1:2,:)=q2(1:2,:)/S2;
T2=[eye(2)/S2,-Mean2(1:2)/S2;0 0 1];

% Construct B-matrix
B = [];
for i = 1:size(q1,2)
   B = [B ; kron(q2(:,i)',CrossOp(q1(:,i)))];
end

%decomposite B 
[u, s, v] = svd(B);
H = v(:,end);
H = T1^(-1)*reshape(H,3,3)*T2;

%Inverse result
H = H^(-1);

end

