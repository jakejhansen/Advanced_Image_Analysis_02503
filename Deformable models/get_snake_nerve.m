function [x, y] = get_snake(x, y, im, im_gray, update_size, disp, alpha, beta, col)
x_old = x;
y_old = y;

try
    mask = poly2mask(x,y, size(im_gray,1), size(im_gray,2));
    %m_in = mean(im(mask == 1));
    %m_out = mean(im(mask == 0));
    
    m_in = 0.36
    m_out = 0.4544
    
    % 5: Compute forces
    
    F_ext = (m_in-m_out)*(2*im-m_in-m_out);
    
    %Round of snake points
    x_r = round(x);
    x_r(x_r < 1) = 1;
    x_r(x_r > size(im_gray,2)) = size(im_gray,2);
    y_r = round(y);
    y_r(y_r < 1) = 1;
    y_r(y_r > size(im_gray,1)) = size(im_gray,1);
    
    %Get values under points
    F_point = [];
    for i = 1:length(x)
        F_point = [F_point, F_ext(y_r(i), x_r(i))];
    end
    
    % 6: Compute Normals
    %Put extra point at start and end to make circular
    
    snake_norms = compute_norms(x,y);
    
    
    %imagesc(im);
    %hold on;
    plot([x, x(1)], [y, y(1)], col, 'linewidth', 3);
    x2 = x;
    y2 = y;
    
    %Update Snake
    for i = 1:size(x,2)
        x(i) = x(i) + update_size*F_point(i)*snake_norms(1,i);
        y(i) = y(i) + update_size*F_point(i)*snake_norms(2,i);
    end
    %plot([x, x(1)], [y, y(1)], 'y');
    
    if disp == true;
        for i = 1:size(x,2)
            plot([x(i),x2(i)],[y(i),y2(i)],'y', 'linewidth',2);
        end
    end
    
    % 7: Regularize circle
    %Refine with alpha and beta
    X = [x ; y]';
    X_new_extend = extended_kernel(X, 0.5, alpha, beta);
    
    x = X_new_extend(:,1)';
    y = X_new_extend(:,2)';
    %plot([x, x(1)], [y, y(1)], 'black');
    
    
    % 8: Redistribute points
    redist = distribute_points([x ; y]');
    x = redist(:,1)';
    y = redist(:,2)';
    
    % remove crossing
    X = [x ; y]';
    X = remove_crossings(X);
    x = X(:,1)';
    y = X(:,2)';
catch e
    x = x_old;
    y = y_old;
    fprintf(1,'The identifier was:\n%s',e.identifier);
    fprintf(1,'There was an error! The message was:\n%s',e.message);
end


end

