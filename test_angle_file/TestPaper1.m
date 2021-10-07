%% 512 patch
clear classes;
mod = py.importlib.import_module('test_single_example');
py.importlib.reload(mod);
filepath = 'defocus_dataset/';
levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
total_run = 10;img_size = 512;
num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
for i=1:13
    file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
end
file{14} = dir([filepath,num2str(47),'/*.tiff']);
dist_focus = 0;
num_predicted = 0;
rng(0);
dist = [];
dist_eachrun = 0;

for num = 1: total_run
    disp(num);
    dist_eachrun = 0;
    for i = 1:55
        choose_level = ceil(13*rand);
        img1 = imread([file{choose_level}(i).folder,'/',file{choose_level}(i).name]);
        img2 = imread([file{choose_level+1}(i).folder,'/',file{choose_level+1}(i).name]);
        actual_level = choose_level;
        %crop into 512*512
        height = size(img1,1);width = size(img1,2);
        x = ceil((height - img_size-1)*rand());y = ceil((width - img_size-1)*rand());
        img1 = img1(x:x+512-1,y:y+512-1);
        img2 = img2(x:x+512-1,y:y+512-1);
        res = py.test_single_example.test_one_image_512_paper1(img1,img2);
        pred = single(res{1});
        cert = single(res{2});
        if any(cert>0.3)
            pred_good = pred(cert>0.3);
            x = [];
            for index = 0:12
                x = [x,length(find(pred_good==index))];
            end
            [a,pred_level] = max(x);
            dist_focus = dist_focus + abs((pred_level-actual_level));
            num_predicted = num_predicted + 1;
        end
        dist_eachrun = dist_eachrun + abs(pred_level-actual_level)*6;
    end
    dist = [dist,dist_eachrun];
end
dist_focus/num_predicted
%dist_focus/num_predicted = 1.03091
%num_predicted = 1100

%% 128 patch
% clear classes;
% mod = py.importlib.import_module('test_single_example');
% py.importlib.reload(mod);
filepath = 'defocus_dataset/';
levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
total_run = 10;img_size = 128;
num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
for i=1:13
    file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
end
file{14} = dir([filepath,num2str(47),'/*.tiff']);
dist_focus = 0;
num_predicted = 0;
rng(0);
dist = [];
dist_eachrun = 0;
dist_eachrun_num = [];
for num = 1: total_run
    disp(num);
    dist_eachrun = 0;
    a= 0;
    for i = 1:55
        
        choose_level = ceil(13*rand);
        img1 = imread([file{choose_level}(i).folder,'/',file{choose_level}(i).name]);
        img2 = imread([file{choose_level+1}(i).folder,'/',file{choose_level+1}(i).name]);
        actual_level = choose_level;
        %crop into 128*128
        height = size(img1,1);width = size(img1,2);
        x = ceil((height - img_size-1)*rand());y = ceil((width - img_size-1)*rand());
        img1 = img1(x:x+img_size-1,y:y+img_size-1);
        img2 = img2(x:x+img_size-1,y:y+img_size-1);
        res = py.test_single_example.test_one_image_128_paper1(img1,img2);
        pred = single(res{1});
        cert = single(res{2});
        if cert > 0.3
            dist_focus = dist_focus + abs((pred-actual_level+1));
            num_predicted = num_predicted + 1;
            a = a+1;
            dist_eachrun = dist_eachrun + abs((pred-actual_level+1))*6;
        end
    end
    dist = [dist,dist_eachrun];
    dist_eachrun_num = [dist_eachrun_num,a];
end
dist_focus/num_predicted
%dist_focus/num_predicted = 1.196222
%num_predicted = 953
        
        
        
        