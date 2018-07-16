clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Pre-Processing MERCK Dataset
% Segment sub-volumes: 1) tumor volumes; 2) non-tumor volumes 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Get the tumor middle slice
% 2. Crop tumor bbox (<70pixels: 70x70 ; >70pixels: resample->70x70)

%% load raw 3d data
data_dir = '../../Data/merck_data/Raw_DATA/3D/ALL';
save_dir = '../../Data/merck_data/';

xlxs_file = '../../Data/merck_data/Raw_DATA/training_tumor_info_new_delineation1.xlsx';
[~,~,info_raw] = xlsread(xlxs_file);

fileID_train = fopen(strcat(save_dir,'dir/','train_list.txt'),'wt');
% fileID_test = fopen(strcat(save_dir,'dir/','test_list.txt'),'wt');

ind_case = 0;
for i = 1:size(info_raw)
    ind_case = ind_case + 1  % count how many CT cases
    
    %% imread data
    % imread CT volume
    filename_raw = cell2mat(info_raw(i,1));
    filename_raw = regexprep(filename_raw, '_slice(\w+)_CT.png', '_CT');
    filename_raw = strsplit(filename_raw, '\');
    filename = cell2mat(filename_raw(end));

    file_dir = strcat(data_dir, '/', filename);
    V_ct = analyze75read(file_dir);
    V_ct_info = analyze75info(file_dir);
    pixel_size_ct = double(V_ct_info.PixelDimensions);
    
    % make sure pixel_size_ct valid
    if size(pixel_size_ct,2) ~= 3
        pixel_size_ct = [0.75, 0.75, 2.5];
    end
        
    V_ct = int16(V_ct);

    % imread Label volume
    filename_raw = cell2mat(info_raw(i,1));
    filename_raw = regexprep(filename_raw, '_slice(\w+)_CT.png', '_label');
    filename_raw = strsplit(filename_raw, '\');
    filename = cell2mat(filename_raw(end));

    file_dir = strcat(data_dir, '/', filename);
    V_label = analyze75read(file_dir);
    V_label_info = analyze75info(file_dir);
    pixel_size_label = double(V_label_info.PixelDimensions);
    
    % make sure pixel_size_label valid
    if size(pixel_size_label,2) ~= 3
        pixel_size_label = [0.75, 0.75, 2.5];
    end

    V_label = int16(V_label);

    %% select middle slice
    % get the middle slice index
    [~,~,z] = ind2sub(size(V_label), find(V_label==1));
    z_set = unique(z);
    z_mid = z_set(1);
    [x,y] = ind2sub(size(V_label(:,:,z_mid)), find(V_label(:,:,z_mid)==1));
    min_x = min(x); max_x = max(x);
    min_y = min(y); max_y = max(y);
    
    x_size = max_x - min_x; 
    y_size = max_y - min_y;
    
    x_size_phy = pixel_size_ct(1) * (max_x - min_x);
    y_size_phy = pixel_size_ct(2) * (max_y - min_y);
    z_size_phy = max(x_size_phy, y_size_phy);
    z_size = z_size_phy/pixel_size_ct(3);
    min_z = z_mid - round(z_size/2); max_z = z_mid + round(z_size/2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1) crop the tumor volume
    if x_size<=60 && y_size<=60  % if tumor<60x60, directly crop 70x70
        x_start = max(1, min_x - (70 - x_size)/2); x_end = min(size(V_ct,1), max_x + (70 - x_size)/2);
        y_start = max(1, min_y - (70 - y_size)/2); y_end = min(size(V_ct,2), max_y + (70 - y_size)/2); 
        z_start = max(1, round(min_z - 0.5 * z_size)); z_end = min(size(V_ct,3), round(max_z + 0.5 * z_size));   

    else     % if tumor>60x60, crop the original size add 15% width
        x_start = max(1, round(min_x - 0.15 * x_size)); x_end = min(size(V_ct,1), round(max_x + 0.15 * x_size));
        y_start = max(1, round(min_y - 0.15 * y_size)); y_end = min(size(V_ct,2), round(max_y + 0.15 * y_size));    
        z_start = max(1, round(min_z - 0.5 * z_size)); z_end = min(size(V_ct,3), round(max_z + 0.5 * z_size)); 

    end
    
    vol_patch = V_ct(x_start:x_end, y_start:y_end, z_start:z_end);
    midimg_patch = V_ct(x_start:x_end, y_start:y_end, z_mid);

    % save the img & label & txt information
    midimg_file_name = strcat(strrep(filename, '_label', '_img_tumor_'), string(z_mid), '.mat');
    vol_file_name = strcat(strrep(filename, '_label', '_vol_tumor_'), string(z_mid), '.mat');

    midimg_save = char(strcat(save_dir, 'image/', midimg_file_name));
    save(midimg_save, 'midimg_patch');
    vol_save = char(strcat(save_dir, 'volume/', vol_file_name));
    save(vol_save, 'vol_patch');
    
    vec_cls = zeros(1,3); 
    vec_cls(cell2mat(info_raw(i,5))) = 1;

    line = char(strcat(vol_file_name, " ", midimg_file_name, ...
        sprintf(' %d', vec_cls), " \r\n"));
    fprintf(fileID_train, line);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2) crop a random non-tumor volume
    TF = randi([0, 1], 1);
    if TF == 1
        % find the random center for cropping
        x_cen = (max_x + min_x)/2; 
        y_cen = (max_z + min_y)/2; 
        z_cen = (max_z + min_z)/2;

        x_ran = randi([round(x_size/2)+5, round(size(V_ct,1)-x_size/2-5)], 1); 
        y_ran = randi([round(y_size/2)+5, round(size(V_ct,2)-y_size/2-5)], 1); 
        z_ran = randi([round(z_size/2)+5, round(size(V_ct,3)-z_size/2-5)], 1);

        d = sqrt(x_size^2 + y_size^2 + (4/3 * z_size)^2);
        while  1.5 * d > sqrt((x_cen - x_ran)^2 + (y_cen - y_ran)^2 + (4/3*(z_cen - z_ran))^2) && ...
                4.5 * d < sqrt((x_cen - x_ran)^2 + (y_cen - y_ran)^2 + (4/3*(z_cen - z_ran))^2)

            x_ran = randi([round(x_size/2)+5, round(size(V_ct,1)-x_size/2-5)], 1); 
            y_ran = randi([round(y_size/2)+5, round(size(V_ct,2)-y_size/2-5)], 1); 
            z_ran = randi([round(z_size/2)+5, round(size(V_ct,3)-z_size/2-5)], 1);

        end

        % crop the volume using random got
        min_x = max(1, x_ran - x_size/2); max_x = min(size(V_ct,1), x_ran + x_size/2);
        min_y = max(1, y_ran - y_size/2); max_y = min(size(V_ct,2), y_ran + y_size/2);
        min_z = max(1, z_ran - z_size/2); max_z = min(size(V_ct,3), z_ran + z_size/2);
        x_size = max_x - min_x; 
        y_size = max_y - min_y;
        z_size = max_z - min_z;
        z_mid = round((max_z + min_z)/2);

        if x_size<=60 && y_size<=60  % if tumor<60x60, directly crop 70x70
            x_start = max(1, min_x - (70 - x_size)/2); x_end = min(size(V_ct,1), max_x + (70 - x_size)/2);
            y_start = max(1, min_y - (70 - y_size)/2); y_end = min(size(V_ct,2), max_y + (70 - y_size)/2); 
            z_start = max(1, round(min_z - 0.5 * z_size)); z_end = min(size(V_ct,3), round(max_z + 0.5 * z_size));   

        else     % if tumor>60x60, crop the original size add 15% width
            x_start = max(1, round(min_x - 0.15 * x_size)); x_end = min(size(V_ct,1), round(max_x + 0.15 * x_size));
            y_start = max(1, round(min_y - 0.15 * y_size)); y_end = min(size(V_ct,2), round(max_y + 0.15 * y_size));    
            z_start = max(1, round(min_z - 0.5 * z_size)); z_end = min(size(V_ct,3), round(max_z + 0.5 * z_size)); 

        end

        vol_patch = V_ct(x_start:x_end, y_start:y_end, z_start:z_end);
        midimg_patch = double(V_ct(x_start:x_end, y_start:y_end, z_mid));

        % save the img & label & txt information
        midimg_file_name = strcat(strrep(filename, '_label', '_img_nontumor_'), string(z_mid), '.mat');
        vol_file_name = strcat(strrep(filename, '_label', '_vol_nontumor_'), string(z_mid), '.mat');

        midimg_save = char(strcat(save_dir, 'image/', midimg_file_name));
        save(midimg_save, 'midimg_patch');
        vol_save = char(strcat(save_dir, 'volume/', vol_file_name));
        save(vol_save, 'vol_patch');

        vec_cls = zeros(1,3); 

        line = char(strcat(vol_file_name, " ", midimg_file_name, ...
            sprintf(' %d', vec_cls), " \r\n"));
        fprintf(fileID_train, line);
        
    end
    
end

fclose(fileID_train);
% fclose(fileID_test);


