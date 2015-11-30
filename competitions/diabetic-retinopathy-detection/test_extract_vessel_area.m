
%% setting enviroment 
clear ; close all;
global curr_dir 
curr_dir= pwd;
addpath(curr_dir);

out_file = 'vessel_area_test.csv';

base_code_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/competitions/diabetic-retinopathy-detection/';
base_data_path = '/Users/gino/Documents/Kaggle/fast-furious/gitHub/fast-furious/dataset/diabetic-retinopathy-detection/';
test_path = [base_data_path 'test/test_1/'];

%% loading labels 
printf('>> reading train labels ... \n');
test = readtable([base_data_path 'sampleSubmission.csv']);
id = test.image;
labels = test.level;

%% files to analize 
test_files = dir(test_path);
printf('>> there are %d images to analize ... \n',length(test_files));
printf('>> there are %d images to analize ... \n',length(test_files));

%% check that for each label in the train set there's a file 
printf('>>> checking .... \n');
id_in_test = arrayfun(@(x) strrep(x.name , '.jpeg', ''), test_files , 'UniformOutput' , 0);
id_in_test( cell2mat(arrayfun(@(x) x.isdir , test_files , 'UniformOutput' , 0)) ) = [];

for i = 1:size(id,1)
    if sum( strcmp(id(i),id_in_test) ) ~= 1
        error('found image id in train set without an image available!');
    end 
end

printf('>>> ALL IDs image in the train set have an image available to process ... \n');

%% for each file etract vessel are and area_ratio and add them to the train table 
test.vessel_area = zeros(size(train,1),1); 
test.vessel_area_ratio = zeros(size(train,1),1); 

for i = 1:size(test,1)
    printf('> processing %s - id:%s - class:%i ...',  [cell2mat(id(i,:)) '.jpeg'] , ...
        cell2mat(id(i,:)) ,  test{i,2} );
    
    img0 = imread([test_path cell2mat(id(i,:)) '.jpeg']);
    [ area , area_ratio ] = get_vessel_area( img0 );
    
    test{i,'vessel_area'} = area;
    test{i,'vessel_area_ratio'} = area_ratio;
    
    printf('.. area:%f - area_ratio:%f \n' , test{i,'vessel_area'} , test{i,'vessel_area_ratio'} );
end

%% save table on disk 
writetable(test,[base_data_path out_file],'Delimiter',',','QuoteStrings',false);