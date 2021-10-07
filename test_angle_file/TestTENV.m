%% 512 patch
clear classes
filepath = 'defocus_dataset/';
levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
total_run = 10;img_size = 512;
num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
for i=1:13
    file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
end
dist_focus = 0;
dist = [];
dist_eachrun = 0;
for num = 1:total_run
    disp(num);
    dist_eachrun = 0;
    for i=1:55
        for j=1:13
            img{j} = imread([file{j}(i).folder,'/',file{j}(i).name]);
        end
        height = size(img{1},1);
        width = size(img{1},2);
        % crop into 512*5 size
        x = ceil((height - img_size-1)*rand());
        y = ceil((width - img_size-1)*rand());
        for j=1:13
            img{j} = img{j}(x:x+img_size-1,y:y+img_size-1);
        end
        score = measure_img(img,'TENV');
        [a,index] = max(score);
        dist_focus = dist_focus + abs(index - 7);
        dist_eachrun = dist_eachrun + abs(index - 7)*6;
    end
    dist = [dist,dist_eachrun];
end

dist_focus/(total_run*55)
% dist_focus/(total_run*55) = 1.31:512; :128
function score = measure_img(img,method)
    score = [];
    method = upper(method);
    if method == 'DCTS' % Energy of Laplacian
        for i = 1:size(img,2)
            score = [score,DCTS(img{i})];
        end
    elseif method == 'TENV'
        for i = 1:size(img,2)
            score = [score,TENV(img{i})];
        end    
    end
    
end
function fm = DCTS(img)
        r=400;
        img_filt = imgaussfilt(img);
        img_DCT = dct2(img_filt);
        L2 = norm(img_DCT);
        value = 0;
        for i=1:r
            for j=1:r
                if i^2+j^2<r^2
                    val = img_DCT(i,j)/L2;
                    value = value + abs(val)*abslog2(val);
                else
                    continue;
                end
            end
        end
        fm = value*-2/(r^2); 
end
function fm = TENV(img)
        Sx = fspecial('sobel');
        Gx = imfilter(double(img), Sx, 'replicate', 'conv');
        Gy = imfilter(double(img), Sx', 'replicate', 'conv');
        G = Gx.^2 + Gy.^2;
        fm = std2(G)^2;
end
function result = abslog2(x)
        if x > 0
        result = log2(x);
        elseif x < 0
            result = log2(-x);
        elseif x == 0
            result = 0;
        end
 end

%% 128 patch
% filepath = 'defocus_dataset/';
% levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
% total_run = 20;img_size = 128;
% num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
% for i=1:13
%     file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
% end
% dist_focus = 0;
% for num = 1:total_run
%     disp(num);
%     for i=1:55
%         for j=1:13
%             img{j} = imread([file{j}(i).folder,'/',file{j}(i).name]);
%         end
%         height = size(img{1},1);
%         width = size(img{1},2);
%         % crop into 128*128 size
%         x = ceil((height - img_size-1)*rand());
%         y = ceil((width - img_size-1)*rand());
%         for j=1:13
%             img{j} = img{j}(x:x+128-1,y:y+128-1);
%         end
%         score = measure_img(img,'DCTS');
%         [a,index] = max(score);
%         dist_focus = dist_focus + abs(index - 7);
%         
%     end
% end
% dist_focus/(total_run*55)
% % dist_focus/(total_run*55) = 1.541818
% function score = measure_img(img,method)
%     score = [];
%     method = upper(method);
%     if method == 'DCTS' % Energy of Laplacian
%         for i = 1:size(img,2)
%             score = [score,DCTS(img{i})];
%         end
%     end
% end
% function fm = DCTS(img)
%         r=100;
%         img_filt = imgaussfilt(img);
%         img_DCT = dct2(img_filt);
%         L2 = norm(img_DCT);
%         value = 0;
%         for i=1:r
%             for j=1:r
%                 if i^2+j^2<r^2
%                     val = img_DCT(i,j)/L2;
%                     value = value + abs(val)*abslog2(val);
%                 else
%                     continue;
%                 end
%             end
%         end
%         fm = value*-2/(r^2); 
% end
% function result = abslog2(x)
%         if x > 0
%         result = log2(x);
%         elseif x < 0
%             result = log2(-x);
%         elseif x == 0
%             result = 0;
%         end
%  end