%%%%%
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;
addpath(curr_dir);

base_code_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/';
base_data_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/';

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
train_files = dir([base_data_path 'train/train/']);
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
img0 = imread([base_data_path 'train/train/' example_0]);
%imshow(img0);
[ area_0 , area_ratio_0 ] = extract_hemorrhages_area( img0 , 'do_plot'); 
printf('>>> area_0:%f - area_ratio_0:%f \n',area_0,area_ratio_0);


%% example class 1 
img1 = imread([base_data_path 'train/train/' example_1]);
%imshow(img1);
[ area_1 , area_ratio_1 ] = extract_hemorrhages_area( img1 , 'do_plot'); 
printf('>>> area_1:%f - area_ratio_1:%f \n',area_1,area_ratio_1);

%% example class 2
img2 = imread([base_data_path 'train/train/' example_2]);
[ area_2 , area_ratio_2 ] = extract_hemorrhages_area( img2 , 'do_plot'); 
printf('>>> area_2:%f - area_ratio_2:%f \n',area_2,area_ratio_2);


%% example class 3 
img3 = imread([base_data_path 'train/train/' example_3]);
[ area_3 , area_ratio_3 ] = extract_hemorrhages_area( img3 , 'do_plot'); 
printf('>>> area_3:%f - area_ratio_3:%f \n',area_3,area_ratio_3);


%% example class 4
img4 = imread([base_data_path 'train/train/' example_4]);
%imshow(img4);
title ([example_4 ' - class 4'], 'interpreter' , 'none' , 'color', [0 0.7 0]);
[ area_4 , area_ratio_4 ] = extract_hemorrhages_area( img4 , 'do_plot'); 
printf('>>> area_4:%f - area_ratio_4:%f \n',area_4,area_ratio_4);

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

%%
myImg = img3;

figure
subplot(3,2,1);
imshow(myImg)
title('Initial Image')

subplot(3,2,2);
imshow(myImg(:,:,3))
title('Blue plane of initial image')

subplot(3,2,3); 
imshow(intrans( intrans(myImg(:,:,3),'stretch'),'log',3))
title('1 trans. in the proc. - default')

subplot(3,2,4);
imshow(intrans( intrans(imadjust(myImg(:,:,3)),'stretch'),'log',3))
title('1 trans. in the proc. - imadjust')

subplot(3,2,5);
imshow(intrans( intrans(histeq(myImg(:,:,3)),'stretch'),'log',3))
title('1 trans. in the proc. - histeq')

subplot(3,2,6);
imshow(intrans( intrans(adapthisteq(myImg(:,:,3)),'stretch'),'log',3))
title('1 trans. in the proc. - adapthisteq')











