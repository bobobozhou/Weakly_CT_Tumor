clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Pre-Processing NIH DeepLesion Data
% 8 tumor type: 
% 1) bone,
% 2) abdomen
% 3) mediastinum
% 4) liver
% 5) lung
% 6) kidney
% 7) soft tissue
% 8) pelvis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load raw 3d data
data_dir = '../../Data/nih_data/Raw_DATA/Images_png/Images_png/';
csv_filename = '../../Data/nih_data/Raw_DATA/DL_info.csv';
save_dir = '../../Data/nih_data/';
% data_dir = './Images_png/';
% csv_filename = 'DL_info.csv';
% save_dir = './nih_data/';

T = readtable(csv_filename);
cls = table2array(T(:, 10));
ind_valid = find(cls~=-1);

CT_filelist = table2array(T(ind_valid, 1));
CT_clslist = cls(ind_valid);
CT_keyimglist = table2array(T(ind_valid, 5));
CT_rangelist = table2array(T(ind_valid, 12));

CT_bboxlist = table2array(T(ind_valid, 7));
CT_RECISTlist = table2array(T(ind_valid, 6));
CT_imgsizelist = table2array(T(ind_valid, 14));

fileID_train = fopen(strcat(save_dir,'dir/','train_list.txt'), 'a+');

for i = 6836:length(CT_filelist)
    i
    
    % imread the volume data
    folder = strrep(CT_filelist(i), ...
        strcat('_', string(num2str(CT_keyimglist(i), '%03.f')),'.png'), '');
    
    range_vol = str2num(char(CT_rangelist(i)));
    size_img = str2num(char(CT_imgsizelist(i)));
    vol = zeros(size_img(1), size_img(2), range_vol(2)-range_vol(1)+1);
    for n = range_vol(1):range_vol(2)
        img_name = char(strcat(data_dir, folder, '/', num2str(n, '%03.f'), '.png'));
        img = double(imread(img_name)) - 32768;
        vol(:,:,n-range_vol(1)+1) = img;
    end
    vol(vol<=-1000)=-1000;
    
    keyimg = CT_keyimglist(i);
    img_key_name = char(strcat(data_dir, folder, '/', num2str(keyimg, '%03.f'), '.png'));
    img_key = double(imread(img_key_name)) - 32768;
    
    % imread and generate the recist image
    img_recist = zeros(str2num(char(CT_imgsizelist(i))));
    
    recist = str2num(char(CT_RECISTlist(i)));
    recist_long = reshape(recist(1:4),2,2)'; recist_long = sortrows(recist_long, 2);
    recist_short = reshape(recist(5:8),2,2)'; recist_short = sortrows(recist_short, 2);
    
    longx_range = linspace(max(1, recist_long(1, 2)), min(recist_long(2, 2), size_img(1)), 1000);
    longy_range = interp1([max(1, recist_long(1, 2)), min(recist_long(2, 2), size_img(1))+0.1], ...
        [max(1, recist_long(1, 1)), min(recist_long(2, 1), size_img(2))+0.1], longx_range);
    
    shortx_range = linspace(max(1, recist_short(1, 2)), min(recist_short(2, 2), size_img(1)), 1000);
    shorty_range = interp1([max(1, recist_short(1, 2)), min(recist_short(2, 2), size_img(1))+0.1], ...
        [max(1, recist_short(1, 1)), min(recist_short(2, 1), size_img(2))+0.1], shortx_range);
    
    img_recist(sub2ind(str2num(char(CT_imgsizelist(i))), round(longx_range), round(longy_range))) = 1;
    img_recist(sub2ind(str2num(char(CT_imgsizelist(i))), round(shortx_range), round(shorty_range))) = 1;
    
    % generate close-loop image from recist
    img_loop = zeros(str2num(char(CT_imgsizelist(i))));
    img_loop = draw_loop(recist_long, recist_short, img_loop);
    if max(img_loop) == 0
        img_loop = img_recist;
    end
    
    %% 1: crop the tumor sub-volume
    bbox = str2num(char(CT_bboxlist(i)));
    size_x = bbox(4) - bbox(2); size_y = bbox(3) - bbox(1);
    x_start = max(1, bbox(2) - 0.15*size_x); x_end = min(size_img(1), bbox(4) + 0.15*size_x);
    y_start = max(1, bbox(1) - 0.15*size_y); y_end = min(size_img(2), bbox(3) + 0.15*size_y);
    
    vol_crop = vol(x_start:x_end, y_start:y_end, :);
    img_crop = img_key(x_start:x_end, y_start:y_end);
    img_recist_crop = img_recist(x_start:x_end, y_start:y_end);
    
    % Grabcut in 3D using RECIST
    L = superpixels3(vol_crop, round(numel(vol_crop)/40), 'Compactness', 0.00001);  % round(numel(vol_crop)/40)
    
    ROI = false(size(vol)); ROI(bbox(2):bbox(4), bbox(1):bbox(3), :) = true;
    ROI = ROI(x_start:x_end, y_start:y_end, :);
    
    foremask = zeros(size(vol)); foremask(:,:,keyimg - range_vol(1) + 1) = imerode(img_loop, strel('diamond', 0));
    foremask = foremask(x_start:x_end, y_start:y_end, :);
    backmask = zeros(size(vol)); backmask(:,:,:) = repmat(~imdilate(img_loop, strel('diamond', 2)), [1 1 size(vol,3)]);
    backmask = backmask(x_start:x_end, y_start:y_end, :);
    
    vol_crop_gau = imgaussfilt3(vol_crop, 0.88);
    mask_crop = double(lazysnapping(vol_crop_gau, L, foremask, backmask));
    
    % post-processing for Grabcut result
    mask_crop = mask_crop | foremask;   % include the foremask
    
    CC = bwconncomp(mask_crop);  % only keep the Connected-Comp overlay with RECIST
    numPixels = zeros(1, length(CC.PixelIdxList));
    for nn = 1:length(CC.PixelIdxList)
        m = zeros(size(mask_crop));
        m(CC.PixelIdxList{nn}) = 1;
        numPixels(nn) = length(find(m .* foremask) == 1);
    end
    [~,idx] = max(numPixels); 
    mask_crop = zeros(size(mask_crop));
    mask_crop(CC.PixelIdxList{idx}) = 1;
    
    mask_crop = fuse_recist_mask(mask_crop, foremask);  % fuse the imerode Recist mask with current mask
       
    se = strel('sphere', 1); mask_crop = imdilate(mask_crop, se);  % morp processing mask
    se = strel('sphere', 6); mask_crop = imclose(mask_crop, se);
    
%     figure(1),
%     subplot(2,3,1), montage(vol_crop_gau,'DisplayRange', [-200 200]);
%     subplot(2,3,2), montage(mask_crop,'DisplayRange', [-1 1]);
%     subplot(2,3,3), montage(foremask,'DisplayRange', [-1 1]);
%     subplot(2,3,4), montage(backmask,'DisplayRange', [-1 1]);
%     subplot(2,3,5), montage(L,'DisplayRange', []);

    % save the information in txt
    filename = strrep(CT_filelist(i), '.png', '');
    vol_file_name = strcat(filename, '_vol_tumor', '.mat');
    mask_file_name = strcat(filename, '_mask_tumor', '.mat');
    save(char(strcat(save_dir, 'volume/', vol_file_name)), 'vol_crop');
    save(char(strcat(save_dir, 'mask/', mask_file_name)), 'mask_crop');
    
    cls = CT_clslist(i);
    cls_vec = zeros(1, 8); cls_vec(cls) = 1;
    
    line = char(strcat(vol_file_name, " ", mask_file_name, ...
        sprintf(' %d', cls_vec), " \r\n"));
    
    fprintf(fileID_train, line);
    
    %% 2: crop the non-tumor sub-volume
    TF = randi([0, 2], 1);
    if TF == 1 || TF == 0
        bbox = str2num(char(CT_bboxlist(i)));
        size_x = bbox(4) - bbox(2); size_y = bbox(3) - bbox(1);
        x_start = max(1, bbox(2) - 0.15*size_x); x_end = min(size_img(1), bbox(4) + 0.15*size_x);
        y_start = max(1, bbox(1) - 0.15*size_y); y_end = min(size_img(2), bbox(3) + 0.15*size_y);
        
        sel = ones(size_img);
        sel(x_start:x_end, y_start:y_end) = 0;
        sel(1:0.18*size_img(1)+1, :) = 0; sel(end-0.18*size_img(1)-1:end, :) = 0;
        sel(:, 1:0.18*size_img(2)+1) = 0; sel(:,end-0.18*size_img(2)-1:end) = 0;
        
        if ~isempty(find(sel==1))
            [x_cen_list, y_cen_list] = find(sel==1);
            ind_cen = randi([1, length(x_cen_list)], 1);
            x_cen = x_cen_list(ind_cen); y_cen = y_cen_list(ind_cen);
            
            x_start = max(1, x_cen - size_x/2); x_end = min(size_img(1), x_cen + size_x/2);
            y_start = max(1, y_cen - size_y/2); y_end = min(size_img(2), y_cen + size_y/2);

            vol_crop = vol(x_start:x_end, y_start:y_end, :);

            % save the information in txt
            filename = strrep(CT_filelist(i), '.png', '');
            vol_file_name = strcat(filename, '_vol_nontumor', '.mat');
            save(char(strcat(save_dir, 'volume/', vol_file_name)), 'vol_crop');

            cls_vec = zeros(1, 8);

            line = char(strcat(vol_file_name, " ", 'NAN.mat', ...
                sprintf(' %d', cls_vec), " \r\n"));

            fprintf(fileID_train, line);
        end
    end

end

fclose(fileID_train);



