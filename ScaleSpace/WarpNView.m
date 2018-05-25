function WarpNView(H,ImL,ImR)



T = maketform('projective',H');
[ImH,XData,YData]=imtransform(ImL,T,'FillValues',[0;0;0]);

Mask=ones(480,640);
MaskH=imtransform(Mask,T,'FillValues',0)>0;

XData=[floor(XData(1)) ceil(XData(2))];
YData=[floor(YData(1)) ceil(YData(2))];
nX=max(size(ImR,2),XData(2))-min(0,XData(1));
nY=max(size(ImR,1),YData(2))-min(0,YData(1));
Origo=[-XData(1) -YData(1)];

M=ones(nY,nX)*1e-3;
M(1:size(ImH,1),1:size(ImH,2))=MaskH+1e-3;
M([Origo(2)+(1:size(ImR,1))],[Origo(1)+(1:size(ImR,2))])=...
    M([Origo(2)+(1:size(ImR,1))],[Origo(1)+(1:size(ImR,2))])+1;
Mask=zeros(nY,nX,3);
Mask(:,:,1)=M;
Mask(:,:,2)=M;
Mask(:,:,3)=M;


Out=zeros(nY,nX,3);

Out(1:size(ImH,1),1:size(ImH,2),:)=double(ImH)./Mask(1:size(ImH,1),1:size(ImH,2),:);

Out([Origo(2)+(1:size(ImR,1))],[Origo(1)+(1:size(ImR,2))],:)=...
Out([Origo(2)+(1:size(ImR,1))],[Origo(1)+(1:size(ImR,2))],:)+...
double(ImR)./Mask([Origo(2)+(1:size(ImR,1))],[Origo(1)+(1:size(ImR,2))],:);

Out=Out/255;
imagesc(Out)
