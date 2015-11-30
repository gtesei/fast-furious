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




%% example class 1 
%img1 = imread([path_train example_1]);
img1 = adapthisteq(imadjust(rgb2gray( imread([path_train example_1]))));
eyeMask = getEyeMask(img1);
img1(~eyeMask) = 0;

points = detectBRISKFeatures(img1);
imshow(img1); hold on;
plot(points);

%%
%corners = detectFASTFeatures(img1);
%corners = detectHarrisFeatures(img1);
corners = detectMinEigenFeatures(img1);
imshow(img1); hold on;
plot(corners);

%%
regions = detectMSERFeatures(img1);
figure; imshow(img1); hold on;
plot(regions, 'showPixelList', true, 'showEllipses', false);

%%
points = detectSURFFeatures(img1);
imshow(img1); hold on;
plot(points);


%% example class 2
img2 = adapthisteq(imadjust(rgb2gray( imread([path_train example_3]))));


%% example class 3 
img3 = adapthisteq(imadjust(rgb2gray( imread([path_train example_3]))));


%% example class 4
img4 = adapthisteq(imadjust(rgb2gray( imread([path_train example_4]))));
%img4 = imread([path_train example_4]);
eyeMask = getEyeMask(img4);
img4(~eyeMask) = 0;
%imshow(img4);

points = detectBRISKFeatures(img4);
imshow(img4); hold on;
plot(points);

%% mean 
num_BRISK_points = points.Count;
mean_X_BRISK_points = mean(points.Location(:,1));
mean_Y_BRISK_points = mean(points.Location(:,2));
mean_Scale_BRISK_points = mean(points.Scale);
mean_Orientation_BRISK_points = mean(points.Orientation);

strongest = points.selectStrongest(ceil(points.Count/10));
num_BRISK_points_str = strongest.Count;
mean_X_BRISK_points_str = mean(strongest.Location(:,1));
mean_Y_BRISK_points_str = mean(strongest.Location(:,2));
mean_Scale_BRISK_points_str = mean(strongest.Scale);
mean_Orientation_BRISK_points_str = mean(strongest.Orientation);

%% corners pints 
%corners = detectFASTFeatures(img4);
%corners = detectHarrisFeatures(img4);
corners = detectMinEigenFeatures(img4);
imshow(img4); hold on;
plot(corners);

%% 
regions = detectMSERFeatures(img4);
figure; imshow(img4); hold on;
plot(regions, 'showPixelList', true, 'showEllipses', false);

%%
points = detectSURFFeatures(img4);
imshow(img4); hold on;
plot(points);



%% remove vessel segmentation 
[ area , area_ratio ] = get_vessel_area( img4 , 'do_plot');

%% remove optic disk 







