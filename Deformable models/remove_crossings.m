function XY = remove_crossings(XY)
%Janus Nørtoft Jensen
%jnje@dtu.dk, December 2016
n = size(XY,1);
[row,col] = line_crossing_check(XY);
XY = XY([1:end 1],:);
for k = 1:numel(row)
    i = row(k); j = col(k);
    f = i+1;
    t = j;
    if ( j-i > n/2 )
        f = j+1;
        t = i+n;
    end
    while ( f < t )
        idF = mod(f-1,n)+1;
        f = f + 1;
        idT = mod(t-1,n)+1;
        t = t - 1;
        tmp = XY(idF,:);
        XY(idF,:) = XY(idT,:);
        XY(idT,:) = tmp;
    end
    XY(end,:) = XY(1,:);
end
XY = XY(1:end-1,:);

function [row, col] = line_crossing_check(XY)
%Janus Nørtoft Jensen
%jnje@dtu.dk, December 2016
n = size(XY,1);
x = XY(:,1);
y = XY(:,2);
xd = x([2:n,1])-x; % dx
yd = y([2:n,1])-y; % dy

a = (x-x').*yd;
b = (y-y').*xd;
c = xd.*yd';

u = (b-a)./(c-c');
h = 0<u & u<1;
intersect_matrix = h & h';
[row,col] = find(triu(intersect_matrix));

