%%%%%
% get_vessel_area 

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

%% process images 
printf('>>> process images ... \n');

%isKey( id_to_class, cellstr(id_pres(ii,:)) )
%values( id_to_class , cellstr(id_pres(ii,:)) )

for ii = 3:size(train_files,1) 
    printf('> processing %s - id:%s - class:%i ...\n', train_files(ii).name , ...
        id_pres(ii,:) , cell2mat(values( id_to_class , cellstr(id_pres(ii,:)) ))  );
    
end 

%% 4 example 

example_0 = [ strtrim(class_0(1,:))  '.jpeg'];
example_1 = [ strtrim(class_1(1,:))  '.jpeg'];
example_2 = [ strtrim(class_2(1,:))  '.jpeg'];
example_3 = [ strtrim(class_3(1,:))  '.jpeg'];
example_4 = [ strtrim(class_4(1,:))  '.jpeg'];


%% example class 0 
img0 = imread([base_data_path 'train/train/' example_0]);
imshow(img0);
title ([example_0 ' - class 0'], 'interpreter' , 'none' , 'color', [0 0.7 0]);
seg_vess = my_vessel_segment(img0);
figure(2), imshow(seg_vess);

%bloodVessels = VesselExtract(rgb2gray(img0),  10);
%figure(3), imshow(bloodVessels);

cc = bwconncomp(seg_vess);
stats = regionprops(cc,'basic');
A = [stats.Area];
area_0 = sum(A);
printf('>> sum area: %f \n' , area_0 );

[ area_0_ck , area_ratio_0 ] = get_vessel_area( img0 );
printf('>> sum area: %f  -- check  - area_ratio: %f \n' , area_0_ck , area_ratio_0  );

%% example class 1 
img1 = imread([base_data_path 'train/train/' example_1]);
imshow(img1);
title ([example_1 ' - class 1'], 'interpreter' , 'none' , 'color', [0 0.7 0]);
seg_vess = my_vessel_segment(img1);
figure, imshow(seg_vess);

cc = bwconncomp(seg_vess);
stats = regionprops(cc,'basic');
A = [stats.Area];
area_1 = sum(A);
printf('>> sum area: %f \n' , area_1 );
[ area_1_ck , area_ratio_1 ] = get_vessel_area( img1 );
printf('>> sum area: %f  -- check  - area_ratio: %f \n' , area_1_ck , area_ratio_1  );


%% example class 2
img2 = imread([base_data_path 'train/train/' example_2]);
imshow(img2);
title ([example_2 ' - class 2'], 'interpreter' , 'none' , 'color', [0 0.7 0]);
seg_vess = my_vessel_segment(img2);
figure, imshow(seg_vess);

cc = bwconncomp(seg_vess);
stats = regionprops(cc,'basic');
A = [stats.Area];
printf('>> sum area: %f \n' , sum(A));

cc = bwconncomp(seg_vess);
stats = regionprops(cc,'basic');
A = [stats.Area];
area_2 = sum(A);
printf('>> sum area: %f \n' , area_2 );

[ area_2_ck , area_ratio_2 ] = get_vessel_area( img2 );
printf('>> sum area: %f  -- check  - area_ratio: %f \n' , area_2_ck , area_ratio_2  );

%% example class 3 
img3 = imread([base_data_path 'train/train/' example_3]);
imshow(img3);
title ([example_3 ' - class 3'], 'interpreter' , 'none' , 'color', [0 0.7 0]);
seg_vess = my_vessel_segment(img3);
figure, imshow(seg_vess);

cc = bwconncomp(seg_vess);
stats = regionprops(cc,'basic');
A = [stats.Area];
area_3 = sum(A);
printf('>> sum area: %f \n' , area_3 );

[ area_3_ck , area_ratio_3 ] = get_vessel_area( img3 );
printf('>> sum area: %f  -- check  - area_ratio: %f \n' , area_3_ck , area_ratio_3  );

%% example class 4
img4 = imread([base_data_path 'train/train/' example_4]);
imshow(img4);
title ([example_4 ' - class 4'], 'interpreter' , 'none' , 'color', [0 0.7 0]);
seg_vess = my_vessel_segment(img4);
figure(2), imshow(seg_vess);

cc = bwconncomp(seg_vess);
stats = regionprops(cc,'basic');
A = [stats.Area];
area_4 = sum(A);
printf('>> sum area: %f \n' , area_4);

[ area_4_ck , area_ratio_4 ] = get_vessel_area( img4 );
printf('>> sum area: %f  -- check  - area_ratio: %f \n' , area_4_ck , area_ratio_4  );

%% print results
printf('>> sum area 0: %f \n' , area_0);
printf('>> sum area 1: %f \n' , area_1);
printf('>> sum area 2: %f \n' , area_2);
printf('>> sum area 3: %f \n' , area_3);
printf('>> sum area 4: %f \n' , area_4);

% sum area 0: 10798.000000 
% sum area 1: 17950.000000 
% sum area 2: 23936.000000 
% sum area 3: 37695.000000 
% sum area 4: 163429.000000 

printf('>> sum area_ratio 0: %f \n' , area_ratio_0);
printf('>> sum area_ratio 1: %f \n' , area_ratio_1);
printf('>> sum area_ratio 2: %f \n' , area_ratio_2);
printf('>> sum area_ratio 3: %f \n' , area_ratio_3);
printf('>> sum area_ratio 4: %f \n' , area_ratio_4);

% sum area_ratio 0: 0.001683 
% sum area_ratio 1: 0.002248 
% sum area_ratio 2: 0.002997 
% sum area_ratio 3: 0.010706 
% sum area_ratio 4: 0.022686 






    




















