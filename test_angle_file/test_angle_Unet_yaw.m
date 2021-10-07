%% 1024
clear classes;
mod = py.importlib.import_module('test_single_example');
py.importlib.reload(mod);
filepath = 'angle_test_compare_1024_new_better_images/561nm';
image_file = dir([filepath,'/*_26.tiff']);
num_test = length(image_file);

ground_truth_yaw = [-0.5349,2.23,0.4487,1.327,1.709,...
    0.8353,0.4095,-1.425,0.6552,-0.6017,...
    -1.359,-1.179,-1.359,-1.654,-2.22,...
    ];

%% prediction
prediction_yaw = [];
for index = 1:num_test
    disp(image_file(index).name);
    image_name1 = image_file(index).name;
    image_name2 = image_file(index).name(1:end-7) + "29.tiff";
    img1 = imread([filepath,'/',image_name1]);
    img2 = imread(strcat(strcat(filepath,'/'),image_name2));
    actual_yaw = ground_truth_yaw(index);
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
    disp(actual_yaw);
    disp(yaw_angle);
end
disp('Done.');

%%
filepath = 'angle_test_compare_1024_new_better_images/561nm_1';
image_file = dir([filepath,'/*_26.tiff']);
num_test = length(image_file);

ground_truth_yaw_1 = [-1.949,-2.374,-0.7207,1.616,-0.2785,...
    -1.85,1.343,2.75,-2.145,0.4586,...
    -1.376,-2.063,0.2057,-0.1474,-2.701];
%%
prediction_yaw_1 = [];
for index = 1:num_test
    disp(image_file(index).name);
    image_name1 = image_file(index).name;
    image_name2 = image_file(index).name(1:end-7) + "29.tiff";
    img1 = imread([filepath,'/',image_name1]);
    img2 = imread(strcat(strcat(filepath,'/'),image_name2));
    actual_yaw = ground_truth_yaw_1(index);
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
    prediction_yaw_1 = [prediction_yaw_1,yaw_angle];
    disp(actual_yaw);
    disp(yaw_angle);
end
disp('Done.');

%%
% for yaw angle the absolute degree error is 0.5040
ground_truth = [ground_truth_yaw,ground_truth_yaw_1];
prediction = [prediction_yaw,prediction_yaw_1];
[truth_sorted,I] = sort(ground_truth);
prediciton_sorted = prediction(I)';
truth_sorted =truth_sorted';