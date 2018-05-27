function snake_norms = compute_norms(x, y)
x_r = round(x);
y_r = round(y);

%Make circulair
x_circ = [x_r(end), x_r, x_r(1)];
y_circ = [y_r(end), y_r, y_r(1)];

snake_norms = [];
for i = 2:size(x_circ,2)-1
    
    %Find vector as differences between points and take normal
    diff_x = x_circ(i+1) - x_circ(i-1);
    diff_y = y_circ(i+1) - y_circ(i-1);
    snake_norm =  [diff_y; -diff_x];
    
    %Normalize to unit normal
    snake_norm = snake_norm./norm(snake_norm,2);
    
    %Append
    snake_norms = [snake_norms, snake_norm];
end
end