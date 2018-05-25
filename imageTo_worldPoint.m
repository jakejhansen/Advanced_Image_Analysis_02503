function P_out = imageTo_worldPoint(xmax,xmin,ymax,ymin)

%Find distance with find_distance function
d = find_distance4032(xmax,xmin,ymax,ymin);

%Locate center of box
x = (xmax+xmin)/2;
y = (ymax+ymin)/2;

%Image Resolution
Rx = 1920;
Ry = 1080;
%Rx = 4032;
%Ry = 3024;

%Focal Length (28 mm) maybe 4 or 8 mm
f = 4 * 10^(-3); 

%Camera Intrinsics
Px = 2.5*10^(-6);
Py = (3+1/3)*10^(-6);
%Px = 1.1905*10^(-6);
%Py = Px;

%Construct transformation matrix
C = [Px , 0 , -(Rx*Px)/2 ; 0 , Py , -(Ry*Py)/2];

%Determine Point at z = f (Focal Length)
P = C * [x;y;1];
P(3) = f;

%Determine scale factor k s.t world-point vector has z = d
%Which is the distance found from find_distance.m
%k = d / f;
k = d / norm(P);

P_out = k*P;
end
