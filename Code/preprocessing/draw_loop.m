function BW = draw_loop(recist_long, recist_short, img)

% create the close loop
l = [recist_long(1,:);
    recist_short(1,:);
    recist_long(2,:);
    recist_short(2,:);
    recist_long(1,:)];

% interpolate with parameter t
t = (1:5)';
v = [l(:,1), l(:,2)];

t_interp = linspace(1,5,200);
v_interp = interp1(t, v, t_interp, 'spline');
x_interp  = v_interp(:,1); y_interp = v_interp(:,2);

% draw on image
BW = roipoly(img, x_interp, y_interp);

end

