%% get x,y,z datapoints
% x = [];
% y = [];
% z = [];
% pred_img = imread('pred_img.tif');
% [height,width] = size(pred_img);
% for i = 1:32:height
%     for j = 1:32:width
%         if pred_img(i,j) == 100
%             continue;
%         else
%             x = [x,i];
%             y = [y,j];
%             z = [z,(pred_img(i,j)*12-6)*6];
%         end   
%     end
% end
%% fit plane
%stem3(x,y,z,'filled','-','LineWidth',0.0,'MarkerSize',2);zlim([-36,36]);xlim([1,2048]);ylim([1,2048]);
% [a,num_points] = size(x);
% s = 0.5*ones(1,num_points);
% scatter3(x,y,z,10,'filled');zlim([-36,36]);xlim([1,2048]);ylim([1,2048]);
% X = [ones(num_points,1),x',y'];
% b = regress(z',X);
% [ROW,COL] = meshgrid(1:2048,1:2048);
% Z = b(1) + b(2)*ROW + b(3)*COL;
% hold on;mesh(ROW,COL,Z,'FaceAlpha',0.05);zlabel('Defocus distance');xlabel('x');ylabel('y')
% %% fit plane 2
% [a,num_points] = size(x);
% s = 5*ones(1,num_points);
% scatter3(x,y,z,10,'filled');zlim([-36,36]);xlim([1,2048]);ylim([1,2048]);xlable('x');ylabel('y')
% X = [x',y'];
% mdlr = fitlm(X,z','Robustopts','off');
% [ROW,COL] = meshgrid(1:2048,1:2048);
% Z = mdlr.Coefficients.Estimate(1) + mdlr.Coefficients.Estimate(2)*ROW + mdlr.Coefficients.Estimate(3) *mdlr.Coefficients.Estimate(3);
% hold on;mesh(ROW,COL,Z,'FaceAlpha',0.05);zlabel('Defocus distance');xlabel('');ylabel('');
%%
function [roll_angle, yaw] = fitplane(pred_img,path)
    x = [];
    y = [];
    z = [];
    [height,width] = size(pred_img);
    % use the Unet
    if path == 1
        for i = 1:8:height
            for j = 1:8:width
                if pred_img(i,j) == 100
                    continue;
                else
                    x = [x,i];
                    y = [y,j];
                    z = [z,(pred_img(i,j)-6)*6];
                end   
            end
        end
        [a,num_points] = size(x);
        X = [ones(num_points,1),x',y'];
        b = regress(z',X);
        %[ROW,COL] = meshgrid(1:2048,1:2048);
        %Z = b(1) + b(2)*ROW + b(3)*COL;
        % calculate roll yaw and defocus
        % roll angle
        z1 = b(1) + b(2)*1 + b(3)*512;
        z2 = b(1) + b(2)*1024 + b(3)*512;
        roll_angle = atan((z1-z2)/((1024-1)*0.65))/pi*180;
        % yaw angle
        z1 = b(1) + b(2)*512 + b(3)*(1);
        z2 = b(1) + b(2)*512 + b(3)*(1024);
        yaw = atan((z2-z1)/(1024*0.65))/pi*180;
    end
    if path == 2
        [height,width] = size(pred_img);
        for i = 1:height
            for j = 1:width
                if pred_img(i,j) == 100
                    continue;
                else
                    x = [x,64 + (i-1)*128];
                    y = [y,64 + (j-1)*128];
                    z = [z,(pred_img(i,j)-6)*6];
                end 
            end
        end
        [a,num_points] = size(x);
        X = [ones(num_points,1),x',y'];
        b = regress(z',X);
        % roll angle
        z1 = b(1) + b(2)*1 + b(3)*512;
        z2 = b(1) + b(2)*1024 + b(3)*512;
        roll_angle = atan((z1-z2)/((1971-100)*0.65))/pi*180;
        % yaw angle
        z1 = b(1) + b(2)*512 + b(3)*(1);
        z2 = b(1) + b(2)*512 + b(3)*(1024);
        yaw = atan((z2-z1)/(1024*0.65))/pi*180;
        
    end
    
end
