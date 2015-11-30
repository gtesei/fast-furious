%%%%%
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;
addpath(curr_dir);

base_code_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/';
base_data_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/';
path_train = [base_data_path 'train/train/'];

%% loading labels 
printf('>> reading labels ... \n')
fid = fopen([base_data_path 'trainLabels.csv'], 'r');
tline = fgetl(fid);
%  Split header
A(1,:) = regexp(tline, '\,', 'split');
%  Parse and read rest of file
ctr = 1;
while(~feof(fid))
if ischar(tline)    
      ctr = ctr + 1;
      tline = fgetl(fid);         
      A(ctr,:) = regexp(tline, '\,', 'split'); 
else
      break;     
end
end
fclose(fid);

id = char(A(2:end,1));
labels = str2num(char(A(2:end,2)));

%% for instance wich class is 10_left
printf('>> for example, class of 10_left == %d \n', labels(strcmp(id , {'10_left'})) );

%% files to analize 
train_files = dir(path_train);
printf('>> there are %d images to analize ... \n',length(train_files));
printf('>> there are %d images to analize ... \n',length(train_files));
printf('>> for example, the image %s is associated to the class %d ... \n',train_files(10).name, ...
    labels(strcmp(id , {strrep(train_files(10).name, '.jpeg', '')})) );

%% now lets partizionate 
class_0 = id(labels == 0,:);
class_1 = id(labels == 1,:);
class_2 = id(labels == 2,:);
class_3 = id(labels == 3,:);
class_4 = id(labels == 4,:);
printf('******************************************\n');
printf('class 0 - num: %i , perc: %f \n' , size(class_0,1) , size(class_0,1) / size(labels,1) );
printf('class 1 - num: %i , perc: %f \n' , size(class_1,1) , size(class_1,1) / size(labels,1) );
printf('class 2 - num: %i , perc: %f \n' , size(class_2,1) , size(class_2,1) / size(labels,1) );
printf('class 3 - num: %i , perc: %f \n' , size(class_3,1) , size(class_3,1) / size(labels,1) );
printf('class 4 - num: %i , perc: %f \n' , size(class_4,1) , size(class_4,1) / size(labels,1) );
printf('******************************************\n');

%% filter on the file in the train set 
printf('>> filtering on the file on the train set ... \n');

% e.g. counts = arrayfun(@(x) numel(x.name), train_files);
id_pres = char(arrayfun(@(x) strrep(x.name , '.jpeg', ''), train_files , 'UniformOutput' , 0));

%is_is_present = cell2mat( arrayfun(@(x) sum(strcmp(x , cellstr(id_pres))), cellstr(id ) , 'UniformOutput' , 0) );

% id_to_class('10003_left') == 0
id_to_class = containers.Map( cellstr(id) ,labels  , 'UniformValues',false );

%% 4 example 

example_0 = [ strtrim(class_0(2,:))  '.jpeg'];
example_1 = [ strtrim(class_1(2,:))  '.jpeg'];
example_2 = [ strtrim(class_2(2,:))  '.jpeg'];
example_3 = [ strtrim(class_3(2,:))  '.jpeg'];
example_4 = [ strtrim(class_4(2,:))  '.jpeg'];


%% example class 0 
img0 = adapthisteq(imadjust(rgb2gray( imread([path_train example_0]))));
%imshow(img0);
redPlane = img0(:,:,1);
redPlane(redPlane<210) = 0;

figure
subplot(1,2,1);
imshow(img0)
title('Initial truecolor image')

subplot(1,2,2);
imshow(redPlane)
title('Filtered image')



%% example class 1 
%img1 = adapthisteq(imadjust(rgb2gray( imread([path_train example_1]))));
img1 = imread([path_train example_1]);
getEyeMask(img1,'perimeter' , 30 , 'do_plot');


%% example class 2
img2 = adapthisteq(imadjust(rgb2gray( imread([path_train example_3]))));

redPlane = img2(:,:,1);
redPlane(redPlane<210) = 0;

figure
subplot(1,2,1);
imshow(img2)
title('Initial truecolor image')

subplot(1,2,2);
imshow(redPlane)
title('Filtered image')


%% example class 3 
img3 = adapthisteq(imadjust(rgb2gray( imread([path_train example_3]))));

redPlane = img3(:,:,1);
redPlane(redPlane<210) = 0;

figure
subplot(1,2,1);
imshow(img3)
title('Initial truecolor image')

subplot(1,2,2);
imshow(redPlane)
title('Filtered image')


%% example class 4
img4 = adapthisteq(imadjust(rgb2gray( imread([path_train example_4]))));
%img4 = imadjust(adapthisteq(rgb2gray( imread([path_train example_4]))));
%imtool(img4);

redPlane = img4(:,:,1);
redPlane( redPlane<210 ) = 0;

figure
subplot(1,2,1);
imshow(img4)
title('Initial truecolor image')

subplot(1,2,2);
imshow(redPlane)
title('Filtered image')


%% remove vessel segmentation 
[ area , area_ratio ] = get_vessel_area( img4 , 'do_plot');

%% remove optic disk 

%% get exudates 

%% summary 
printf('****************************\n');
printf('area_0:%f - area_ratio_0:%f \n' , area_0, area_ratio_0 );
printf('area_1:%f - area_ratio_1:%f \n' , area_1, area_ratio_1 );
printf('area_2:%f - area_ratio_2:%f \n' , area_2, area_ratio_2 );
printf('area_3:%f - area_ratio_3:%f \n' , area_3, area_ratio_3 );
printf('area_4:%f - area_ratio_4:%f \n' , area_4, area_ratio_4 );
printf('****************************\n');

% area_0:324.000000 - area_ratio_0:0.000046 
% area_1:96.000000 - area_ratio_1:0.000020 
% area_2:249980.000000 - area_ratio_2:0.064980 <<<<<
% area_3:1098992.000000 - area_ratio_3:0.295488 <<<<<<<
% area_4:177659.000000 - area_ratio_4:0.025051 



