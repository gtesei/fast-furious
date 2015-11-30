
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;
addpath(curr_dir);

base_code_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/';
base_data_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/';
train_image_path = [base_data_path 'train/train/'];

sample_size = 2000; 

out_file = ['img_' int2str(sample_size) '_adjust_eq.csv'];

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
img_mat = zeros(size(train,1),3751);
img_mat(:,1) = train.level;

for ii = 1:length(sample)
    i = sample(ii);
    printf('> processing %s - id:%s - class:%i ...\n',  [cell2mat(id(i,:)) '.jpeg'] , ...
        cell2mat(id(i,:)) ,  train{ii,2} );
    
    
    % process image  
    img0 = imread([base_data_path 'train/train/' cell2mat(id(i,:)) '.jpeg']);
    img0 = adapthisteq(imadjust(rgb2gray(img0)));
    %img0 = imresize(img0(:,:,2),[50 75]);
    img0 = imresize(img0,[50 75]);
    img_mat(ii,2:end) = img0(:);
end

%% save table on disk 
csvwrite([base_data_path out_file],img_mat);