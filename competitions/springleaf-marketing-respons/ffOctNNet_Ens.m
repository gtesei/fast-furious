
%% making enviroment 
menv;
DEBUG = 0;

%% id 
%% id = 'layer1_dataProcdoc_proc_2_modffOctNNet_tuneTRUE.csv';
%% id = 'layer1_dataProcdoc_pcaElbow_modffOctNNet_tuneTRUE.csv';  
%% id = 'layer1_dataProcdoc_PC100_modffOctNNet_tuneTRUE.csv'; 
%%id = 'layer1_dataProcdoc_PCA95Var_modffOctNNet_tuneTRUE.csv'; 
id = 'layer3_dataProcdoc_default_modffOctNNet_tuneTRUE.csv';

%% test_id , ytrain 
printf('>>> loading test_id and ytrain ... \n');
fflush(stdout);
test_id = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/test_id4octave.csv']);
ytrain = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Ytrain4octave.csv']);

fprintf('>>> loaded test_id: %i rows \n' , length(test_id));
fprintf('>>> loaded ytrain: %i rows -- [ytrain == 0]:%i  -- [ytrain == 1]:%i \n' , length(ytrain) , sum(ytrain==0) , sum(ytrain==1) );    
fflush(stdout);

%% train / test set 
printf('>>> loading train / test set ... \n');
%% Xtrain = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtrain_docproc2.csv']);
%% Xtest = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtest_docproc2.csv']);

%%Xtrain = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtrain_docproc2_pca_elbow_4octave.csv']);
%%Xtest = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtest_docproc2_pca_elbow_4octave.csv']);  

%%Xtrain = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtrain_docproc2_PC100_4octave.csv']);
%%Xtest = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtest_docproc2_PC100_4octave.csv']);

%%Xtrain = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtrain_docproc2_pca_95var_4octave.csv']);
%%Xtest = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtest_docproc2_pca_95var_4octave.csv']);

Xtrain = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtrain_ens_l12_4octave.csv']);
Xtest = dlmread([curr_dir '/dataset/springleaf-marketing-respons/elab/Xtest_ens_l12_4octave.csv']);

fprintf('>>> loaded Xtrain: %i rows - %i columns \n' , size(Xtrain,1) , size(Xtrain,2));
fprintf('>>> loaded Xtest : %i rows - %i columns \n' , size(Xtest,1) , size(Xtest,2));
fflush(stdout);

if (DEBUG == 1) 
  printf('>>> DEBUG mode activated: making small data sets ... \n');
  fflush(stdout);
  Xtrain = Xtrain(1:100,:);
  ytrain = ytrain(1:100);
  Xtest = Xtest(1:100,:);
  test_id = test_id(1:100);
end 

%% tune / train / predict / ensemble
printf('>>> tuning / training / predicting / ensembling  ... \n');
fflush(stdout);
tic();  
[predTrain , predTest , p_opt_AUC, h_opt_AUC, lambda_opt_AUC, AUC, grid] = ...
          findOptPAndHAndLambda_kfold_ensembleAndPredict(Xtrain, ytrain, Xtest,...
                                featureScaled = 0 , scaleFeatures = 0 , ...
                                p_vec = [] , ...
                                h_vec = [1 2 3] , ...
                                lambda_vec = [0 0.001 0.01 0.1 1 5] , ...
                                verbose = 1, doPlot=0 , ...
                                initGrid = [] , initStart = -1 , ...
                                iter = 200 , iter_pred = 200 , ...
                                regression = 0 , num_labels = 1 , k = 4);
tm = toc();

%% submission 
submission = [test_id predTest];
dlmwrite(['/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/springleaf-marketing-respons/ensembles/pred_ensemble_3/' id], submission );

%% best tune 
bestTune = [tm AUC p_opt_AUC h_opt_AUC lambda_opt_AUC];
dlmwrite(['/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/springleaf-marketing-respons/ensembles/best_tune_3/' id], bestTune );

%% ensemble 
assemble = [ predTrain ; predTest ];
assemble_id = (1:length(assemble))';
ensemble = [assemble_id assemble];
dlmwrite(['/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/springleaf-marketing-respons/ensembles/ensemble_3/' id], ensemble );
