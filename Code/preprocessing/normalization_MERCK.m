clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Normalize Merck Dataset
% Merck DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load raw 3d data
data_dir = '../../Data/merck_data/Raw_DATA/3D/ALL';
files = dir(data_dir);
save_dir = '../../Data/merck_data/Raw_DATA/3D_normalized/ALL';

ind_case = 0;
for i = 3:4:length(files)
    ind_case = ind_case + 1  % count how many CT cases
    
    %% imread data
    % imread CT volume
    filename = files(i).name;
    if contains(filename, '_CT.hdr')
        file_dir = strcat(files(i).folder, '/',...
            strrep(filename, '_CT.hdr','_CT'));
        V_ct = double(analyze75read(file_dir));
        V_ct_info = analyze75info(file_dir);
        pixel_size_ct = V_ct_info.PixelDimensions;
        
        % make sure pixel_size_ct valid
        if size(pixel_size_ct,2) ~= 3
            pixel_size_ct = [0.75, 0.75, 2.5];
        end
        
        % pipeline for normalize/shift to (>0) HU value
        if min(V_ct(:)) < -1000   % if there is outside (negative padding)
            V_ct(V_ct<-1000) = -3000;  % replace with -3000
        end
        
        min_vol = min(V_ct(:));
        if min_vol <= -1001 && min(V_ct(V_ct > min_vol)) < -800
            V_ct = V_ct + 1000;
        else
            if min(V_ct(V_ct > min_vol)) < -800
                V_ct = V_ct + 1000;
            end
        end
        V_ct(V_ct<-500) = 0;
        
        % interpolate ct volume & save
        size_org = size(V_ct);
        size_new = round(size_org .* (pixel_size_ct ./ [0.75, 0.75, 1]));
        [X,Y,Z] = meshgrid(1:size_org(2), 1:size_org(1), 1:size_org(3));
        [Xq,Yq,Zq] = meshgrid(linspace(1, size_org(2), size_new(2)), ...
            linspace(1, size_org(1), size_new(1)), ...
            linspace(1, size_org(3), size_new(3)));
        
        V_ct_new = interp3(X, Y, Z, V_ct, Xq, Yq, Zq, 'spline');
        V_ct_new(V_ct_new<0) = 0;
        
        fprintf('max= %0.1f; min= %0.1f;', max(V_ct_new(:)), min(V_ct_new(:)));
    
        % save
        vol_file_name = filename;
        vol_save = char(strcat(save_dir, '/', vol_file_name)); 
        writeanalyze(int16(permute(V_ct_new, [2 1 3])), size_new, vol_save, [0.75, 0.75, 1])
        
    end

    % imread Label volume
    filename = files(i+2).name;
    if contains(filename, '_label.hdr')
        file_dir = strcat(files(i+2).folder, '/',...
            strrep(filename, '_label.hdr','_label'));
        V_label = double(analyze75read(file_dir));
        V_label_info = analyze75info(file_dir);
        pixel_size_label = V_label_info.PixelDimensions;
        
        % make sure pixel_size_label valid
        if size(pixel_size_label,2) ~= 3
            pixel_size_label = [0.75, 0.75, 2.5];
        end
        
        % interpolate ct volume & save
        size_org = size(V_label);
        size_new = round(size_org .* (pixel_size_label ./ [0.75, 0.75, 1]));
        [X,Y,Z] = meshgrid(1:size_org(2), 1:size_org(1), 1:size_org(3));
        [Xq,Yq,Zq] = meshgrid(linspace(1, size_org(2), size_new(2)), ...
            linspace(1, size_org(1), size_new(1)), ...
            linspace(1, size_org(3), size_new(3)));
        
        V_label_new = interp3(X, Y, Z, V_label, Xq, Yq, Zq, 'nearest');
    
        % save
        vol_file_name = filename;
        vol_save = char(strcat(save_dir, '/', vol_file_name)); 
        writeanalyze(uint8(permute(V_label_new, [2 1 3])), size_new, vol_save, [0.75, 0.75, 1])
        
    end
    
end


