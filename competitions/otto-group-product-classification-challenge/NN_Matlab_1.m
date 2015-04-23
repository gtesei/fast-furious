
%% making enviroment 
menv;
prefix_fn = [curr_dir '/dataset/otto-group-product-classification-challenge/'];


%% loading data 
disp('|--> loading train file ... \n');
train_raw = csvread([prefix_fn 'oct_train_encoded.csv']); 
disp('|--> loading test file ... \n');
test_raw = csvread([prefix_fn 'oct_test_encoded.csv']); 
disp('|--> loading y file ... \n');
y = csvread([prefix_fn 'oct_y_encoded.csv']); 

%% making data structures 
% labels = unique(y);
% yClust = zeros(length(labels),length(labels));
% yP = (y * y');
% for  i = 1:length(labels) 
%     yClust = (yP == i);
%     if (i == 3) 
%         break;
%     end
% end

%% neural nets 

disp('|--> making data for matlab ... \n');

train_mat = train_raw';
test_mat = test_raw';

y1 = (y==1); 
y2 = (y==2);
y3 = (y==3);
y4 = (y==4);
y5 = (y==5);
y6 = (y==6);
y7 = (y==7);
y8 = (y==8);
y9 = (y==9);

y_mat = [y1' ; y2' ; y3' ; y4' ; y5'; y6'; y7'; y8'; y9'];

clear train_raw  test_raw; 
clear y1 y2 y3 y4 y5 y6 y7 y8 y9 y; 

%% nntool data 
idx = randperm(size(train_mat,2));
train_mat = train_mat(:,idx);
y_mat = y_mat(:,idx);

%train_mat = train_mat(:,1:1000);
%y_mat = y_mat(:,1:1000);

%% lunch net 
disp('|--> training ... \n');

nn_advanced_script;

disp('|--> predicting ... \n');

pred = net(test_mat);
pred = pred';

%% storing results 
disp('|--> storing results  ... \n');
csvwrite([prefix_fn 'pred_nn_matlab_2.csv'] , pred)











