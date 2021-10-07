Input_img_path = "8_17_2021_cochlea/561nm";
Output_img_path = "defocus_dataset_angle";
for i = 1:51
    file_path = Output_img_path + '/' + num2str(i);
    if ~exist(file_path)
        mkdir(file_path);
    end
end

imgs_file = dir(Input_img_path + '/*.tiff');

for i = 1: length(imgs_file)
    disp(i);
    img_path = [imgs_file(i).folder , '/' , imgs_file(i).name];
    img = imread(img_path);
    img = img(1024-511:1024+512,1024-511:1024+512);
    % get the defocus level
    k1 = strfind(img_path,'_');
    k2 = strfind(img_path,'.tiff');
    defocus_level = img_path(k1(end)+1:k2-1);
    %disp(defocus_level);
    imwrite(img,Output_img_path + '/' + num2str(defocus_level)+ '/' + imgs_file(i).name);
end






