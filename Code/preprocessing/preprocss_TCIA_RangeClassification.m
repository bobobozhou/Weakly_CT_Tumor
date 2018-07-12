% clear
% clc
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % MATLAB Script for Pre-Processing Public Dataset
% % NSCLC Radiogenomics: The Cancer Imaging Archive (TCIA) Public Access
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 1. Find the tumor middle slice
% % 2. Get the upper & bottom slices with positive and negative sameple
% 
% %% load raw 3d data
% data_dir = '../../Data_RangeClassification/public_data/Raw_DATA/3D';
% files = dir(data_dir);
% save_dir = '../../Data_RangeClassification/public_data/';
% 
% fileID_all = fopen(strcat(save_dir,'dir/','all_list.txt'),'wt');
% 
% ind_case = 0;
% for i = 3:4:length(files)
%     ind_case = ind_case + 1  % count how many CT cases
%     
%     %% imread data
%     % imread CT volume
%     filename = files(i).name;
%     if contains(filename, '_CT.hdr')
%         file_dir = strcat(files(i).folder, '/',...
%             strtok(filename, '_CT.hdr'),'_CT');
%         V_ct = analyze75read(file_dir);
%     end
%     V_ct = uint16(V_ct);
% 
%     % imread Label volume
%     filename = files(i+2).name;
%     if contains(filename, '_label.hdr')
%         file_dir = strcat(files(i+2).folder, '/',...
%             strtok(filename, '_label.hdr'),'_label');
%         V_label = analyze75read(file_dir);
%     end
%     V_label = uint16(V_label);
% 
%     %% select middle slice & crop the tumor
%     % get the middle slice index
%     [~,~,z] = ind2sub(size(V_label), find(V_label==1));
%     [ min_z, max_z ] = middle_cut(z);
%     z_mid = round((max_z + min_z)/2);
%     
%     for zz = max(min_z - round((max_z-min_z)/2), 1):min(max_z + round((max_z-min_z)/2), size(V_ct, 3))        
%         img_tumor = V_ct(:, :, zz);
%         img_mid_tumor = V_ct(:, :, z_mid);
%         mask_mid_tumor = V_label(:, :, z_mid);
% 
%         % save the img & label & txt information
%         img_file_name = strcat(strtok(filename, '_label.hdr'), '_img_', string(zz), '.png');
%         img_mid_file_name = strcat(strtok(filename, '_label.hdr'), '_img_', string(z_mid), '.png');
%         mask_mid_file_name = strcat(strtok(filename, '_label.hdr'), '_mask_', string(z_mid), '.png');
%         
%         img_save = char(strcat(save_dir, 'image/', img_file_name));
%         imwrite(img_tumor, img_save);
%         img_mid_save = char(strcat(save_dir, 'image/', img_mid_file_name));
%         imwrite(img_mid_tumor, img_mid_save);
%         mask_mid_save = char(strcat(save_dir, 'mask/', mask_mid_file_name));
%         imwrite(mask_mid_tumor, mask_mid_save);
%         
%         dis_to_center = sprintf('%0.2f', abs(zz - z_mid)/(max_z - min_z));
%         
%         if zz <= max_z && zz >= min_z
%             line = char(strcat(string(ind_case), " ", string(dis_to_center), " ", ...
%                 img_file_name, " ", img_mid_file_name, " ", mask_mid_file_name, " ", ...
%                 " 1\r\n"));
%             fprintf(fileID_all, line);
%         else
%             line = char(strcat(string(ind_case), " ", string(dis_to_center), " ", ...
%                 img_file_name, " ", img_mid_file_name, " ", mask_mid_file_name, " ", ...
%                 " 0\r\n"));
%             fprintf(fileID_all, line);
%         end
%             
% %         figure(1),
% %         subplot(1,3,1); I=imread(img_save); imshow(I,[-1000 3000]);
% %         subplot(1,3,2); I=imread(img_mid_save); imshow(I,[-1000 3000]);
% %         subplot(1,3,3); I=imread(mask_mid_save); imshow(I,[0 1]);
% %         drawnow;
% 
%     end
%     
% end
% 
% fclose(fileID_all);

%% randomly split all the data
all = [];
fid = fopen(strcat(save_dir,'dir/','all_list.txt'));
tline = fgetl(fid);
while ischar(tline)
    tline = fgetl(fid);
    all = [all; string(tline)];
end
all = all(1:end-1);

ind = randperm(length(all));

% traning list
fileID_train = fopen(strcat(save_dir,'dir/','train_list.txt'),'wt');

ind_train = ind(1:round(0.8 * length(all)));
train_list = all(ind_train);
for i = 1:length(train_list)
    line = char(strcat(train_list(i), "\r\n"));
    fprintf(fileID_train, line);
end

fclose(fileID_train);

% testing list
fileID_test = fopen(strcat(save_dir,'dir/','test_list.txt'),'wt');

ind_test = ind(round(0.8 * length(all)):length(all));
test_list = all(ind_test);
for i = 1:length(test_list)
    line = char(strcat(test_list(i), "\r\n"));
    fprintf(fileID_test, line);
end

fclose(fileID_test);


