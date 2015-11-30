
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;
addpath(curr_dir);

base_code_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/';
base_data_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/';
train_image_path = [base_data_path 'train/train/'];

sample_size = 2000; 

out_file = ['feat_gen_' int2str(sample_size) '.csv'];

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

%% sampling 
sample = datasample(1:size(train,1),sample_size,'Replace',false);
train = train(sample,:);


%% for each file etract vessel are and area_ratio and add them to the train table 
train.num_BRISK_points = zeros(size(train,1),1); 
train.mean_X_BRISK_points = zeros(size(train,1),1); 
train.mean_Y_BRISK_points = zeros(size(train,1),1); 
train.mean_Orientation_BRISK_points = zeros(size(train,1),1); 

train.num_BRISK_points_str = zeros(size(train,1),1); 
train.mean_X_BRISK_points_str = zeros(size(train,1),1); 
train.mean_Y_BRISK_points_str = zeros(size(train,1),1); 
train.mean_Scale_BRISK_points_str = zeros(size(train,1),1); 
train.mean_Orientation_BRISK_points_str = zeros(size(train,1),1); 

for ii = 1:length(sample)
    i = sample(ii);
    printf('> processing %s - id:%s - class:%i ...',  [cell2mat(id(i,:)) '.jpeg'] , ...
        cell2mat(id(i,:)) ,  train{ii,2} );
    
    
    % read & adjiust image 
    img0 = adapthisteq(imadjust(rgb2gray( imread([base_data_path 'train/train/' cell2mat(id(i,:)) '.jpeg']) )));
    
    % mask 
    eyeMask = getEyeMask(img0);
    img0(~eyeMask) = 0;
    
    % feat
    points = detectBRISKFeatures(img0);
    
    % feature extraction  
    train{ii,'num_BRISK_points'} = points.Count;
    train{ii,'mean_X_BRISK_points'} = mean(points.Location(:,1));
    train{ii,'mean_Y_BRISK_points'} = mean(points.Location(:,2));
    train{ii,'mean_Scale_BRISK_points'} = mean(points.Scale);
    train{ii,'mean_Orientation_BRISK_points'} = mean(points.Orientation);

    strongest = points.selectStrongest(ceil(points.Count/10));
    train{ii,'num_BRISK_points_str'} = strongest.Count;
    train{ii,'mean_X_BRISK_points_str'} = mean(strongest.Location(:,1));
    train{ii,'mean_Y_BRISK_points_str'} = mean(strongest.Location(:,2));
    train{ii,'mean_Scale_BRISK_points_str'} = mean(strongest.Scale);
    train{ii,'mean_Orientation_BRISK_points_str'} = mean(strongest.Orientation);
    
    printf('..\n');disp(train(ii,:));
end

%% save table on disk 
writetable(train,[base_data_path out_file],'Delimiter',',','QuoteStrings',false);