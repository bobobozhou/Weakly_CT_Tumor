folder_dir = '../../Data/merck_data/volume';
files = dir(folder_dir);

for i = 3:length(files)
    filename = files(i).name;
    file_dir = strcat(files(i).folder, '/', filename);
    
    load(file_dir);
    vol_patch = permute(vol_patch, [3,1,2]);
    
    save(file_dir, 'vol_patch')
    
end