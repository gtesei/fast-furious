#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

find_par_mode = 1;

trainFile = "train_NO_NA_oct.zat"; 
%trainFile = "train_v2_NA_CI_oct.zat";
testFile = "test_v2_NA_CI_oct.zat";   


%%% 1) FEATURES ENGINEERING 
printf("|--> FEATURES BUILDING ...\n");

data = dlmread([curr_dir "/dataset/loan_default/" trainFile]); %%NA clean in R
if (find_par_mode)
  [m,n] = size(data);
  rand_indices = randperm(m);
  data = data(rand_indices,:);
  %%data = data(1:10000,:);
endif

y_loss = data(:,end);
y_def = (y_loss > 0) * 1 + (y_loss == 0)*0; 

%%Xcat = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
Xcont = [data(:,522) data(:,523) data(:,270)];  

%%%%%%%%%%%%% merge test set for encoding categorical features 
%%data = [];
# data = dlmread([curr_dir "/dataset/loan_default/" testFile]); %%NA clean in R
# Xcat_test = [data(:,3) data(:,6) data(:,768) data(:,769)]; 
# Xcat_tot = [Xcat;Xcat_test]; 
# data = [];
# [Xcat_totE,map,offset] = encodeCategoricalFeatures(Xcat_tot);
# Xcat_totE = []; Xcat_tot = [];
%%%%%%%%%%%%%%

%[XcatE,map,offset] = encodeCategoricalFeatures(Xcat,map,offset);
[Xcont,mu,sigma] = featureNormalize(Xcont);

X = [Xcont];
y = [y_def y_loss];

X = [ones(size(X,1),1) X];
[m,n] = size(X)
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),0.90); 

ytrain_def = ytrain(:,1);
ytrain_loss = ytrain(:,2);

yval_def = yval(:,1);
yval_loss = yval(:,2);

%%%%%%%%%%%%%%%%%%%%%% STEP FORWARD PROCEDUERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
initFeat = [270 522 523];
trashFeat = [1 32 33 34 36 37 670 692 693 694 728 756];

eIdx = size(data,2) -1;

bestMAE = -1; 

bestREG_MAE = -1;
bestP = -1; 
bestRTHETA = -1; 

bestF1 = -1; 
bestEpsilon = -1;
bestACC = -1;
bestALLTHETA = -1;

REG_MAEs = [];
F1s = [];
ACCs = [];
MAEs= [];

%%% initial conditions 
addedFeat_reg = initFeat;
addedFeat_class = initFeat;
%%%%%%%%%%%%%%%%%%%%%% 

for idx = 0:eIdx  

  printf("|--> Processing feature index = %i (0 = first iteration) ...\n",idx);

  %%%%%%%%%%% skipping not applicable features 
  if (  sum(idx == initFeat) > 0 )
    printf("|-. feature belonging to initial set: continue ... ");
    continue; 
  endif 

  if ( sum(idx == trashFeat) > 0 )
    printf("|-. trash feature: continue ... ")
    continue; 
  endif

  %%%%%%%%%%%%%%%%% feature re-building 
  Xcont_reg = [];
  Xcont_class = [];

  for k = 1:length(addedFeat_reg)
    Xcont_reg = [Xcont_reg data(:, addedFeat_reg(k) )];
  endfor 

  for k = 1:length(addedFeat_class)
    Xcont_class = [Xcont_class data(:, addedFeat_class(k) )];
  endfor 

  %%%% initial cond
  Xtrain_reg = Xtrain;
  Xval_reg = Xval;
  Xtrain_class = Xtrain;
  Xval_class = Xval; 
  %%%%%%%%%%%%%%%%%%%% trying feature idx 
  if ( idx != 0)
    
    Xcont_reg = [Xcont_reg data(:,idx)];
    Xcont_class = [Xcont_class data(:,idx)];
    
  endif
  
    [Xcont_reg,mu,sigma] = featureNormalize(Xcont_reg);
    [Xcont_class,mu,sigma] = featureNormalize(Xcont_class);

    X_reg = [Xcont_reg];
    X_class = [Xcont_class];

    y = [y_def y_loss];

    X_reg = [ones(size(X_reg,1),1) X_reg];
    X_class = [ones(size(X_class,1),1) X_class];
    
    m = size(y,1);

    rand_indices = randperm(m);
    [Xtrain_reg,ytrain,Xval_reg,yval] = splitTrainValidation(X_reg(rand_indices,:),y(rand_indices,:),0.90); 
    [Xtrain_class,ytrain,Xval_class,yval] = splitTrainValidation(X_class(rand_indices,:),y(rand_indices,:),0.90);

    ytrain_def = ytrain(:,1);
    ytrain_loss = ytrain(:,2);

    yval_def = yval(:,1);
    yval_loss = yval(:,2);

  %%endif 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  %%%%%%%%%%%%%%%%%%%%%%% LOSS REGRESSOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%% 3) FINDING BEST PARAMS LOSS REGRESSOR 
  printf("|--> FINDING BEST PARAMS LOSS REGRESSOR  ...\n");
  [p_opt,J_opt] = findOptP_RegLin(Xtrain_reg, ytrain_loss, Xval_reg, yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
  [reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain_reg, ytrain_loss, Xval_reg, yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);

  %% --> model parameters: p_opt , reg_lambda_opt
  REGpars = [p_opt reg_lambda_opt];
  printf("|--> OPTIMAL LINEAR REGRESSOR PARAMS -->  opt. number of polinomial degree (p_opt) = %i , opt. lambda = %f\n",p_opt,reg_lambda_opt);
  %%dlmwrite ('REGpars.zat', REGpars);

  %% --> performance
  %%[Xtrain_poly,mu,sigma] = treatContFeatures(Xtrain_reg,p_opt);
  Xtrain_poly = polyFeatures(Xtrain,p_opt); 
  rtheta = trainLinearReg(Xtrain_poly, ytrain_loss, reg_lambda_opt, 400);
  %%[Xval_poly,mu,sigma] = treatContFeatures(Xval_reg,p_opt,1,mu,sigma);
  Xval_poly = polyFeatures(Xval,p_opt);
  pred_loss = predictLinearReg(Xval_poly,rtheta);
  pred_loss = (pred_loss < 0) .* 0 + (pred_loss > 100) .* 100 +  (pred_loss >= 0 & pred_loss <= 100) .*  pred_loss;
  [mae_reg] = MAE(pred_loss, yval_loss);
  printf("|-> trained loss regressor. MAE on cross validation set = %f  \n",mae_reg);

  %%%%%%%%%%%%%%%%%%%%%%%% DEFAULT  CLASSIFIER  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  i = 95;
  %%%%%%%%%%%%% BOOTSTRAP %%%%%%%%%
  %%%% making training set 50% 0' & 50% 1'
  id0 = find(ytrain_loss == 0);
  id1 = find(ytrain_loss > 0);
  printf("|--> 0/1 ratio:: %i / 100 .... found id0 - %i , id1- %i ... making equal length ...\n",i,length(id0),length(id1));
  r01 = floor( length(id1) * i / 100 ); 
  id0 = id0(1:r01,:);
  printf("|--> made length(id0) == %i , length(id1) == %i ...\n",length(id0),length(id1));
  Xtrain_boot = [Xtrain_class(id1,:); Xtrain_class(id0,:)];
  ytrain_boot = [ytrain(id1,:); ytrain(id0,:)];

  %%% shuffle 
  rand_indices = randperm(size(Xtrain_boot,1));
  Xtrain_boot = Xtrain_boot(rand_indices,:);
  ytrain_boot = ytrain_boot(rand_indices,:);

  ytrain_def_boot = ytrain_boot(:,1);
  ytrain_loss_boot = ytrain_boot(:,2);   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  lambda_log = 0;
  [all_theta] = oneVsAll(Xtrain_boot, ytrain_def_boot, 1, lambda_log ,800);
  pval = sigmoid(Xval_class * all_theta' );
  [epsilon F1] = selectThreshold(yval_def, pval);
  fprintf("\n found bestEpsilon: %f       F1:%f      \n",epsilon,F1);
  pred_log = (pval > epsilon);
  acc_log = mean(double(pred_log == yval_def)) * 100;
  fprintf("\n Logistic classifier - training set accuracy (p=%i,lambda=%f): %f\n", 1,lambda_log,acc_log);

  pred_comb_log = (pred_log == 0) .* 0 + (pred_log == 1) .* pred_loss;
  [mae_log] = MAE(pred_comb_log, yval_loss);
  printf("|-> COMBINED PREDICTION --> MAE on cross validation set = %f  (mae_reg=%f) (F1=%f) (ACC=%f) \n",mae_log, mae_reg, F1, acc_log);
  
  %%%%%%%%%%%%%%% update %%%%%%%%%%%%%%%%%%
  if (  idx == 0  )
    printf("|-! first iteration -  performance with feature index == %i (mae=%f) (mae_reg=%f) (F1=%f) (ACC=%f) !\n",idx,mae_log,mae_reg,F1,acc_log);
    bestREG_MAE = mae_reg;
    bestP = p_opt;
    bestRTHETA = rtheta;   
    bestMAE = mae_log; 
    bestF1 = F1; 
    bestEpsilon = epsilon; 
    bestACC = acc_log;
    bestALLTHETA = all_theta; 
    ACCs = [ACCs acc_log]; 
    F1s = [F1s F1];
    MAEs = [MAEs mae_log];
    REG_MAEs = [REG_MAEs mae_reg];
    %%addedFeat_reg = [addedFeat_reg idx];
    %%addedFeat_class = [addedFeat_class idx];

  elseif ( (mae_reg < bestREG_MAE) & (F1 <= bestF1) )
    printf("|-! regressor ONLY improvment -  performance with feature index == %i (mae=%f) (mae_reg=%f) (F1=%f) (ACC=%f) !\n",idx,mae_log,mae_reg,F1,acc_log);
    bestREG_MAE = mae_reg;
    bestP = p_opt; 
    bestRTHETA = rtheta;
    %%bestF1 = F1;
    %%bestEpsilon = epsilon;
    %%bestACC = acc_log;
    %%bestALLTHETA = all_theta;
    ACCs = [ACCs acc_log];
    F1s = [F1s F1];
    MAEs = [MAEs mae_log];
    REG_MAEs = [REG_MAEs mae_reg];
    addedFeat_reg = [addedFeat_reg idx];
    %%addedFeat_class = [addedFeat_class idx];
  elseif ( (mae_reg < bestREG_MAE) & (F1 > bestF1) )  
    printf("|-! classifier ONLY improvment -  performance with feature index == %i (mae=%f) (mae_reg=%f) (F1=%f) (ACC=%f) !\n",idx,mae_log,mae_reg,F1,acc_log);
    %%bestREG_MAE = mae_reg;
    %%bestP = p_opt;
    %%bestRTHETA = rtheta;  
    bestF1 = F1;                                                                                                                                                                                      
    bestEpsilon = epsilon;                                                                                                                                                                                
    bestACC = acc_log;
    bestALLTHETA = all_theta;                                                                                                                                                                       
    ACCs = [ACCs acc_log];
    F1s = [F1s F1];
    MAEs = [MAEs mae_log];
    REG_MAEs = [REG_MAEs mae_reg];
    %%addedFeat_reg = [addedFeat_reg idx];
    addedFeat_class = [addedFeat_class idx]; 
  elseif ( (mae_reg < bestREG_MAE) & (F1 > bestF1) )
    printf("|-! classifier and regressor improvment -  performance with feature index == %i (mae=%f) (F1=%f) (ACC=%f) !\n",idx,mae_log,F1,acc_log);
    bestREG_MAE = mae_reg;
    bestP = p_opt;
    bestRTHETA = rtheta;                                                                                                                                                    
    bestF1 = F1;
    bestEpsilon = epsilon;
    bestACC = acc_log;
    bestALLTHETA = all_theta;
    ACCs = [ACCs acc_log];
    F1s = [F1s F1];
    MAEs = [MAEs mae_log];
    REG_MAEs = [REG_MAEs mae_reg];
    addedFeat_reg = [addedFeat_reg idx];                                                                                                                                                                  
    addedFeat_class = [addedFeat_class idx];
  endif  

  if (mae_log > bestMAE)
    bestMAE = mae_log; 
  endif 

endfor 

printf("|-. model tuned with %i features for regressor \n",length(addedFeat_reg));disp(addedFeat_reg);
printf("|-. model tuned with %i features for classifier \n",length(addedFeat_class));disp(addedFeat_class);
printf("|-. model performances: (bestMAE=%f) (bestREG_MAE=%f) (bestF1=%f)  (bestACC=%f) \n" , bestMAE , bestREG_MAE , bestF1 , bestACC );

printf("|-> prediction on test set ...  \n");

testFile = "test_v2_NA_CI_oct.zat";
data = dlmread([curr_dir "/dataset/loan_default/" testFile]);

Xcont_reg = [];
Xcont_class = [];

for k = 1:length(addedFeat_reg)
  Xcont_reg = [Xcont_reg data(:, addedFeat_reg(k) )];
endfor

for k = 1:length(addedFeat_class)
  Xcont_class = [Xcont_class data(:, addedFeat_class(k) )];
endfor

%%Xcont = [data(:,522) data(:,523) data(:,270)];
[Xcont_reg,mu,sigma] = featureNormalize(Xcont_reg);
[Xcont_class,mu,sigma] = featureNormalize(Xcont_class);

Xtest_reg = [Xcont_reg];
Xtest_reg = [ones(size(Xtest_reg,1),1) Xtest_reg];

Xtest_class = [Xcont_class];
Xtest_class = [ones(size(Xtest_class,1),1) Xtest_class];

####### loss prediction 
%%[Xtest_poly,mu,sigma] = treatContFeatures(Xtest_reg,bestP);
Xtest_poly = polyFeatures(Xtest_reg,bestP); 
predtest_loss = predictLinearReg(Xtest_poly,bestRTHETA);
predtest_loss = (predtest_loss < 0) .* 0 + (predtest_loss > 100) .* 100 +  (predtest_loss >= 0 & predtest_loss <= 100) .*  predtest_loss;

####### default prediction
ptval = sigmoid(Xtest_class * bestALLTHETA' );
predtest_log = (ptval > bestEpsilon);

##### combinata 
predtest_comb = (predtest_log == 0) .* 0 + (predtest_log == 1) .* predtest_loss;

ids = data(:,1);
sub_comb = [ids predtest_comb];

dlmwrite ('sub_comb_log.csv', sub_comb,",");
