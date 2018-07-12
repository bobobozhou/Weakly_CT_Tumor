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
fileID_test = fopen(strcat(save_dir,'dir/','test_list.txt'),'wt');

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
    V_ct = uint16(V_ct);

    % imread Label volume
    filename = files(i+2).name;
    if contains(filename, '_label.hdr')
        file_dir = strcat(files(i+2).folder, '/',...
            strtok(filename, '_label.hdr'),'_label');
        V_label = analyze75read(file_dir);
    end
    V_label = uint16(V_label);

    %% select middle slice & crop the tumor
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
        z_start = round(min(z) - 0.15 * z_size); z_end = round(max(z) + 0.15 * z_size);   

    else     % if tumor>60x60, crop the original size add 15% width
        x_start = round(min(x) - 0.15 * x_size); x_end = round(max(x) + 0.15 * x_size);
        y_start = round(min(y) - 0.15 * y_size); y_end = round(max(y) + 0.15 * y_size);    
        z_start = round(min(z) - 0.15 * z_size); z_end = round(max(z) + 0.15 * z_size); 

    end
    
    vol_patch_tumor = V_ct(x_start:x_end, y_start:y_end, z_start:z_end);
    midimg_patch_tumor = V_ct(x_start:x_end, y_start:y_end, z_mid);

    % save the img & label & txt information
    midimg_file_name = strcat(strtok(filename, '_label.hdr'), '_img_', string(z_mid), '.png');
    vol_file_name = strcat(strtok(filename, '_label.hdr'), '_vol_', string(z_mid), '.mat');

    midimg_save = char(strcat(save_dir, 'image/', midimg_file_name));
    imwrite(midimg_patch_tumor, midimg_save);
    vol_save = char(strcat(save_dir, 'volume/', vol_file_name));
    save(vol_save, 'vol_patch_tumor');
%     imwrite(vol_patch_tumor, vol_save);

    line = char(strcat(vol_file_name, ...
        " 1 0 0 \r\n"));
    fprintf(fileID_train, line);
    
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % 1) crop a random non-tumor volume
%     if x_size<=60 && y_size<=60  % if tumor<60x60, directly crop 70x70
%         x_start = min_x - (70 - x_size)/2; x_end = max_x + (70 - x_size)/2;
%         y_start = min_y - (70 - y_size)/2; y_end = max_y + (70 - y_size)/2; 
%         z_start = round(min(z) - 0.15 * z_size); z_end = round(max(z) + 0.15 * z_size);   
% 
%     else     % if tumor>60x60, crop the original size add 15% width
%         x_start = round(min(x) - 0.15 * x_size); x_end = round(max(x) + 0.15 * x_size);
%         y_start = round(min(y) - 0.15 * y_size); y_end = round(max(y) + 0.15 * y_size);    
%         z_start = round(min(z) - 0.15 * z_size); z_end = round(max(z) + 0.15 * z_size); 
% 
%     end
%     
%     vol_patch_tumor = V_ct(x_start:x_end, y_start:y_end, z_start:z_end);
%     midimg_patch_tumor = V_ct(x_start:x_end, y_start:y_end, z_mid);
% 
%     % save the img & label & txt information
%     midimg_file_name = strcat(strtok(filename, '_label.hdr'), '_img_', string(z_mid), '.png');
%     vol_file_name = strcat(strtok(filename, '_label.hdr'), '_vol_', string(z_mid), '.mat');
% 
%     midimg_save = char(strcat(save_dir, 'image/', midimg_file_name));
%     imwrite(midimg_patch_tumor, midimg_save);
%     vol_save = char(strcat(save_dir, 'volume/', vol_file_name));
%     save(vol_save, 'vol_patch_tumor');
% %     imwrite(vol_patch_tumor, vol_save);
% 
%     line = char(strcat(vol_file_name, " ", midimg_file_name, ...
%         " 1 0 0 \r\n"));
%     fprintf(fileID_train, line);
    
end

fclose(fileID_train);
fclose(fileID_test);


