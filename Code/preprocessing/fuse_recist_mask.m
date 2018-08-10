function mask_new = fuse_recist_mask(mask, foremask)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% find the slice range and mask
[~,~,z] = ind2sub(size(mask), find(mask==1));
z_mask_range = unique(z);

% find the recist slice and mask
[~,~,z] = ind2sub(size(foremask), find(foremask==1));
z_mask_key = unique(z);
[x, y] = ind2sub(size(foremask(:,:,z_mask_key)), find(foremask(:,:,z_mask_key)==1));
min_x = min(x); max_x = max(x);
min_y = min(y); max_y = max(y);
cen_x = round((min(x) + max(x))/2); cen_y = round((min(y) + max(y))/2);
size_x = max_x - min_x; size_y = max_y - min_y;

img_sub = foremask(min_x:max_x,min_y:max_y,z_mask_key);

% determine how much shrink (minimal 4x4 mask)
mm = [4,4];

a = linspace(mm(1)/size_x, 1, abs(min(z_mask_range) - z_mask_key) + 1);
b = fliplr(linspace(mm(1)/size_x, 1, abs(z_mask_key - max(z_mask_range)) + 1));
factors_x = [a(1:end-1), 1, b(2:end)];

a = linspace(mm(2)/size_y, 1, abs(min(z_mask_range) - z_mask_key) + 1);
b = fliplr(linspace(mm(2)/size_y, 1, abs(z_mask_key - max(z_mask_range)) + 1));
factors_y = [a(1:end-1), 1, b(2:end)];

% resize image according to the factors
for i = min(z_mask_range):max(z_mask_range)
    size_x_new = max(1, round(size_x * factors_x(i-min(z_mask_range)+1)));
    size_y_new = max(1, round(size_y * factors_y(i-min(z_mask_range)+1)));
    img_sub_rs = imresize(img_sub, [size_x_new, size_y_new]);
    
    img = zeros(size(foremask(:,:,z_mask_key)));
    x_start = max(1, cen_x-floor(size_x_new/2) + 1); x_end = min(x_start + size_x_new - 1, size(img,1));
    y_start = max(1, cen_y-floor(size_y_new/2) + 1); y_end = min(y_start + size_y_new - 1, size(img,2));
    img(x_start:x_end, y_start:y_end) = img_sub_rs(1:x_end-x_start+1, 1:y_end-y_start+1);
    
    mask(:,:,i) = mask(:,:,i) | img;
    
end


mask_new = mask;
end

