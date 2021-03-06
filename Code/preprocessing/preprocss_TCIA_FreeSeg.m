clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Pre-Processing Public Dataset
% NSCLC Radiogenomics: The Cancer Imaging Archive (TCIA) Public Access
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Get the tumor middle slice
% 2. Crop tumor bbox (<70pixels: 70x70 ; >70pixels: resample->70x70)

%% load raw 3d data
data_dir = '../../Data/public_data/Raw_DATA/3D_normalized';
files = dir(data_dir);
save_dir = '../../Data/public_data/';

fileID_train = fopen(strcat(save_dir,'dir/','train_list.txt'),'wt');
% fileID_test = fopen(strcat(save_dir,'dir/','test_list.txt'),'wt');

ind_case = 0;
for i = 3:4:length(files)
    ind_case = ind_case + 1  % count how many CT cases
    
    %% imread data
    % imread CT volume
    filename = files(i).name;
    if contains(filename, '_CT.hdr')
        file_dir = strcat(files(i).folder, '/',...
            strtok(filename, '_CT.hdr'),'_CT');
        V_ct = analyze75read(file_dir);
    end
    V_ct = int16(V_ct);

    % imread Label volume
    filename = files(i+2).name;
    if contains(filename, '_label.hdr')
        file_dir = strcat(files(i+2).folder, '/',...
            strtok(filename, '_label.hdr'),'_label');
        V_label = analyze75read(file_dir);
    end
    V_label = int16(V_label);

    %% select middle slice
    % get the middle slice index
    [~,~,z] = ind2sub(size(V_label), find(V_label==1));
    [ min_z, max_z ] = middle_cut(z);
    z_mid = round((max_z + min_z)/2);
    [x,y] = ind2sub(size(V_label(:,:,z_mid)), find(V_label(:,:,z_mid)==1));
    min_x = min(x); max_x = max(x);
    min_y = min(y); max_y = max(y);
    
    x_size = max_x - min_x; 
    y_size = max_y - min_y;
    z_size = max_z - min_z;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1) crop the tumor volume
    if x_size<=60 && y_size<=60  % if tumor<60x60, directly crop 70x70
        x_start = min_x - (70 - x_size)/2; x_end = max_x + (70 - x_size)/2;
        y_start = min_y - (70 - y_size)/2; y_end = max_y + (70 - y_size)/2; 
        z_start = round(min_z - 0.15 * z_size); z_end = round(max_z + 0.15 * z_size);   

    else     % if tumor>60x60, crop the original size add 15% width
        x_start = round(min_x - 0.15 * x_size); x_end = round(max_x + 0.15 * x_size);
        y_start = round(min_y - 0.15 * y_size); y_end = round(max_y + 0.15 * y_size);    
        z_start = round(min_z - 0.15 * z_size); z_end = round(max_z + 0.15 * z_size); 

    end
    
    vol_patch = V_ct(x_start:x_end, y_start:y_end, z_start:z_end);
    midimg_patch = V_ct(x_start:x_end, y_start:y_end, z_mid);

    % save the img & label & txt information
    midimg_file_name = strcat(strrep(filename, '_label.hdr', '_img_tumor_'), string(z_mid), '.mat');
    vol_file_name = strcat(strrep(filename, '_label.hdr', '_vol_tumor_'), string(z_mid), '.mat');

    midimg_save = char(strcat(save_dir, 'image/', midimg_file_name));
    save(midimg_save, 'midimg_patch');
    vol_save = char(strcat(save_dir, 'volume/', vol_file_name));
    save(vol_save, 'vol_patch');
    
    vec_cls = zeros(1,3); 
    vec_cls(1) = 1;

    line = char(strcat(vol_file_name, " ", midimg_file_name, ...
        sprintf(' %d', vec_cls), " \r\n"));
    fprintf(fileID_train, line);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2) crop a random non-tumor volume
    % find the random center for cropping
    x_cen = (max_x + min_x)/2; 
    y_cen = (max_z + min_y)/2; 
    z_cen = (max_z + min_z)/2;
    
    x_ran = randi([round(x_size/2)+5, round(size(V_ct,1)-x_size/2-5)], 1); 
    y_ran = randi([round(y_size/2)+5, round(size(V_ct,2)-y_size/2-5)], 1); 
    z_ran = randi([round(z_size/2)+5, round(size(V_ct,3)-z_size/2-5)], 1);
    
    d = sqrt(x_size^2 + y_size^2 + (4/3 * z_size)^2);
    while  1.5 * d > sqrt((x_cen - x_ran)^2 + (y_cen - y_ran)^2 + (4/3*(z_cen - z_ran))^2) && ...
            8 * d < sqrt((x_cen - x_ran)^2 + (y_cen - y_ran)^2 + (4/3*(z_cen - z_ran))^2)
            
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
        z_start = max(1, round(min_z - 0.15 * z_size)); z_end = min(size(V_ct,3), round(max_z + 0.15 * z_size));   

    else     % if tumor>60x60, crop the original size add 15% width
        x_start = max(1, round(min_x - 0.15 * x_size)); x_end = min(size(V_ct,1), round(max_x + 0.15 * x_size));
        y_start = max(1, round(min_y - 0.15 * y_size)); y_end = min(size(V_ct,2), round(max_y + 0.15 * y_size));    
        z_start = max(1, round(min_z - 0.15 * z_size)); z_end = min(size(V_ct,3), round(max_z + 0.15 * z_size)); 

    end
    
    vol_patch = V_ct(x_start:x_end, y_start:y_end, z_start:z_end);
    midimg_patch = double(V_ct(x_start:x_end, y_start:y_end, z_mid));

    % save the img & label & txt information
    midimg_file_name = strcat(strrep(filename, '_label.hdr', '_img_nontumor_'), string(z_mid), '.mat');
    vol_file_name = strcat(strrep(filename, '_label.hdr', '_vol_nontumor_'), string(z_mid), '.mat');

    midimg_save = char(strcat(save_dir, 'image/', midimg_file_name));
    save(midimg_save, 'midimg_patch');
    vol_save = char(strcat(save_dir, 'volume/', vol_file_name));
    save(vol_save, 'vol_patch');
    
    vec_cls = zeros(1,3);

    line = char(strcat(vol_file_name, " ", midimg_file_name, ...
        sprintf(' %d', vec_cls), " \r\n"));
    fprintf(fileID_train, line);
    
end

fclose(fileID_train);
% fclose(fileID_test);


