
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;
addpath(curr_dir);

out_file = 'vessel_area_train.csv';

base_code_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/';
base_data_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/';
train_image_path = [base_data_path 'train/train/'];

%% loading labels 
printf('>> reading train labels ... \n');
train = readtable([base_data_path 'trainLabels.csv']);
id = train.image;
labels = train.level;

%% files to analize 
train_files = dir(train_image_path);
printf('>> there are %d images to analize ... \n',length(train_files));
printf('>> there are %d images to analize ... \n',length(train_files));
printf('>> for example, the image %s is associated to the class %d ... \n',train_files(10).name, ...
    labels(strcmp(id , {strrep(train_files(10).name, '.jpeg', '')})) );

%% check that for each label in the train set there's a file 
printf('>>> checking .... \n');
id_in_train = arrayfun(@(x) strrep(x.name , '.jpeg', ''), train_files , 'UniformOutput' , 0);
id_in_train( cell2mat(arrayfun(@(x) x.isdir , train_files , 'UniformOutput' , 0)) ) = [];

for i = 1:size(id,1)
    if sum( strcmp(id,id_in_train(i)) ) ~= 1
        error('found image id in train set without an image available!');
    end 
end

printf('>>> ALL IDs image in the train set have an image available to process ... \n');


%% for each file etract vessel are and area_ratio and add them to the train table 
train.vessel_area = zeros(size(train,1),1); 
train.vessel_area_ratio = zeros(size(train,1),1); 

for i = 1:size(train,1)
    printf('> processing %s - id:%s - class:%i ...',  [cell2mat(id(i,:)) '.jpeg'] , ...
        cell2mat(id(i,:)) ,  train{i,2} );
    
    img0 = imread([base_data_path 'train/train/' cell2mat(id(i,:)) '.jpeg']);
    [ area , area_ratio ] = get_vessel_area( img0 );
    
    train{i,'vessel_area'} = area;
    train{i,'vessel_area_ratio'} = area_ratio;
    
    printf('.. area:%f - area_ratio:%f \n' , train{i,'vessel_area'} , train{i,'vessel_area_ratio'} );
end

%% save table on disk 
writetable(train,[base_data_path out_file],'Delimiter',',','QuoteStrings',false);