%% 512 patch
% clear classes;
% mod = py.importlib.import_module('test_single_example');
% py.importlib.reload(mod);
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
dist_eachrun_num = [];
for num = 1: total_run
    disp(num);
    dist_eachrun = 0;
    if num > 1
        disp(['total_round: ',num2str(num),' result: ',num2str(dist_focus/num_predicted)]);
    end
    a = 0;
    for i = 1:55
        disp(['index of images: ',num2str(i),' number of predicted: ',num2str(num_predicted),' total_round: ',num2str(num)]);
        choose_level = ceil(13*rand);
        img1 = imread([file{choose_level}(i).folder,'/',file{choose_level}(i).name]);
        img2 = imread([file{choose_level+1}(i).folder,'/',file{choose_level+1}(i).name]);
        actual_level = choose_level;
        %crop into 512*512
        height = size(img1,1);width = size(img1,2);
        x = ceil((height - img_size-1)*rand());y = ceil((width - img_size-1)*rand());
        img1 = img1(x:x+512-1,y:y+512-1);
        img2 = img2(x:x+512-1,y:y+512-1);
        res = py.test_single_example.test_one_image_512(img1,img2);
        result = single(res);
        
        x = [];
        for index = 0:12
            x = [x,length(find(result==index))];
        end
        if sum(x) > 40000
            [b,pred_level] = max(x);
            dist_focus = dist_focus + abs((pred_level-actual_level));
            num_predicted = num_predicted + 1;
            a = a+1;
            dist_eachrun = dist_eachrun + abs(pred_level-actual_level)*6;
            %disp(['error: ',num2str(dist_focus/num_predicted)]);
        end
        
    end
    dist = [dist,dist_eachrun];
    dist_eachrun_num = [dist_eachrun_num,a];
end
dist_focus/num_predicted
%dist_focus/num_predicted = 0.944272445820433,
%num_predicted = 368


%% 128 patch
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
dist_eachrun_num = [];
for num = 1: total_run
    disp(num);
    dist_eachrun = 0;
    a = 0;
    if num > 1
        disp(['total_round: ',num2str(num),' result: ',num2str(dist_focus/num_predicted)]);
    end
    
    for i = 1:55
        disp(['index of images: ',num2str(i),' number of predicted: ',num2str(num_predicted),' total_round: ',num2str(num)]);
        choose_level = ceil(13*rand);
        img1 = imread([file{choose_level}(i).folder,'/',file{choose_level}(i).name]);
        img2 = imread([file{choose_level+1}(i).folder,'/',file{choose_level+1}(i).name]);
        actual_level = choose_level;
        %crop into 512*512
        height = size(img1,1);width = size(img1,2);
        x = ceil((height - img_size-1)*rand());y = ceil((width - img_size-1)*rand());
        img1 = img1(x:x+512-1,y:y+512-1);
        img2 = img2(x:x+512-1,y:y+512-1);
        res = py.test_single_example.test_one_image_512(img1,img2);
        result = single(res);
        for ii = 1:4
            for jj = 1:4
                pred_small = result(128*(ii-1)+1:128*(ii-1)+128,128*(jj-1)+1:128*(jj-1)+128);
                x = [];
                for index = 0:12
                    x = [x,length(find(pred_small==index))];
                end
                if sum(x)> 2500
                    [b,pred_level] = max(x);
                    dist_focus = dist_focus + abs((pred_level-actual_level));
                    num_predicted = num_predicted + 1;
                    a = a+1;
                    dist_eachrun = dist_eachrun + abs(pred_level-actual_level)*6;
                end
            end
        end
        
    end
    dist = [dist,dist_eachrun];
    dist_eachrun_num = [dist_eachrun_num,a];
end
dist_focus/num_predicted
%dist_focus/num_predicted = 
%num_predicted = 

%%
if any(result<100)
    disp();
end