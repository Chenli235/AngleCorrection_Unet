%% 1024
clear classes;
mod = py.importlib.import_module('test_single_example');
py.importlib.reload(mod);
filepath = 'angle_test_compare_roll_1024/561nm';
image_file = dir([filepath,'/*_26.tiff']);
num_test = length(image_file);
ground_truth_roll = [0.69711,1.202,1.3899,2.4988,2.5655,...
    -1.6797,-1.3533,-1.278,-1.2096,-1.3274,...
    -1.2993,1.7299,0.035629,1.5466,-0.23942,...
    0.31858,2.2683,1.8054,-0.97804,-0.18204,...
    -1.6949,-1.6887,-1.0846,1.9611,1.9406,...
    -1.6318,1.6538,1.0545,1.9794,2.2039]';
ground_truth_yaw = [-1.5247,2.7481,1.3533,2.3326,2.5732,...
    1.7652,0.43395,2.6783,-3.2301,0.90704,...
    1.0197,0,-0.62371,2.173,1.2798,...
    -2.7652,0.3378,2.2444,2.1564,0.31474,...
    -0.34627,-1.7324,-0.89796,-0.89796,-0.73433,...
    0.25368,0,0,-2.4697,0.85903]';
disp(num_test);

%%
prediction_roll = [];
for index = 1:num_test
    disp(image_file(index).name);
    image_name1 = image_file(index).name;
    image_name2 = image_file(index).name(1:end-7) + "29.tiff";
    img1 = imread([filepath,'/',image_name1]);
    img2 = imread(strcat(strcat(filepath,'/'),image_name2));
    actual_roll = ground_truth_roll(index);
    % get inference from network
    pred_img = zeros(1024);
    for row = 1:2
        for col = 1:2
            img1_small = img1(512*(row-1)+1:512*(row-1)+512,512*(col-1)+1:512*(col-1)+512);
            img2_small = img2(512*(row-1)+1:512*(row-1)+512,512*(col-1)+1:512*(col-1)+512);
            res = py.test_single_example.test_one_image_512(img1_small,img2_small);
            result = single(res);
            pred_img(512*(row-1)+1:512*(row-1)+512,512*(col-1)+1:512*(col-1)+512) = result;
        end
    end
    x = [];y=[];z=[];
    for i = 1:8:1024
        for j = 1:8:1024
            if pred_img(i,j) == 100
                continue;
            else
                x = [x,i];
                y = [y,j];
                z = [z,-(pred_img(i,j)-6)*6];
            end
        end
    end
    [a,num_points] = size(x);
    X = [ones(num_points,1),x',y'];
    b = regress(z',X);
    z1 = b(1) + b(2)*1 + b(3)*512;
    z2 = b(1) + b(2)*1024 + b(3)*512;
    roll_angle = atan((z2-z1)/(1024-1)*0.65)/pi*180;
    prediction_roll = [prediction_roll,roll_angle];
    disp(roll_angle);
end
disp('done.');

%% get ascending array for roll
% for roll angle the absolute degree error is 0.2205
[truth_roll_sorted,I] = sort(ground_truth_roll);
prediciton_roll_sorted = prediction_roll(I)';
x = 1:30;

%% get ascending array for yaw
% for yaw angle the absolute degree error is 1.3276
[truth_yaw_sorted,I] = sort(ground_truth_yaw);
prediction_yaw_sorted = prediction_yaw(I)';
%%
filepath = 'angle_test_compart_yaw_1024/561nm';
image_file = dir([filepath,'/*_26.tiff']);
num_test = length(image_file);
prediction_yaw = [];
for index = 1:num_test
    disp(image_file(index).name);
    image_name1 = image_file(index).name;
    image_name2 = image_file(index).name(1:end-7) + "29.tiff";
    img1 = imread([filepath,'/',image_name1]);
    img2 = imread(strcat(strcat(filepath,'/'),image_name2));
    actual_roll = ground_truth_roll(index);
    % get inference from network
    pred_img = zeros(1024);
    for row = 1:2
        for col = 1:2
            img1_small = img1(512*(row-1)+1:512*(row-1)+512,512*(col-1)+1:512*(col-1)+512);
            img2_small = img2(512*(row-1)+1:512*(row-1)+512,512*(col-1)+1:512*(col-1)+512);
            res = py.test_single_example.test_one_image_512(img1_small,img2_small);
            result = single(res);
            pred_img(512*(row-1)+1:512*(row-1)+512,512*(col-1)+1:512*(col-1)+512) = result;
        end
    end
    x = [];y=[];z=[];
    for i = 1:8:1024
        for j = 1:8:1024
            if pred_img(i,j) == 100
                continue;
            else
                x = [x,i];
                y = [y,j];
                z = [z,-(pred_img(i,j)-6)*6];
            end
        end
    end
    [a,num_points] = size(x);
    X = [ones(num_points,1),x',y'];
    b = regress(z',X);
    z1 = b(1) + b(2)*512 + b(3)*1;
    z2 = b(1) + b(2)*512 + b(3)*1024;
    yaw_angle = atan((z2-z1)/(1024*0.65))/pi*180;
    prediction_yaw = [prediction_yaw,yaw_angle];
    disp(yaw_angle);
end
disp('done.');