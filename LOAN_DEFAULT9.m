#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

REGRESSOR_BOOTSTRAP = 0; 

trainFile = "train_NO_NA_oct.zat";
%trainFile = "train_NO_NA_oct_10K.zat"; 
%trainFile = "train_v2_NA_CI_oct.zat";

testFile = "test_v2_NA_CI_oct.zat";   
%testFile = "train_NO_NA_oct_10K.zat";  


bestMAE = -1; 
bestMAE_P = -1;
bestMAE_RTHETA = -1;
bestMAE_Epsilon = -1;
bestMAE_ALLTHETA = -1;
bestMAE_init_perm = -1; 

bestMAE_REG = -1;
bestMAE_F1 = -1;
bestMAE_ACC = -1;


%%% 1) FEATURES ENGINEERING 
%printf("|--> FEATURES BUILDING ...\n");

data = dlmread([curr_dir "/dataset/loan_default/" trainFile]); %%NA clean in R
data_test = dlmread([curr_dir "/dataset/loan_default/" testFile]);

for (in = 1:1000)
printf("|--> iteration # %i  ..... FEATURES BUILDING ...\n",in);

[m,n] = size(data);
rand_indices = randperm(m);
data = data(rand_indices,:);
%%data = data(1:10000,:);

%y_loss = data(:,end);
%y_def = (y_loss > 0) * 1 + (y_loss == 0)*0; 

%FEAT_REG = [270   522   523     3    30    55   100   103   142]; 
%FEAT_CLASS = [270   522   523     2     3     9    19    68   135   142];
%FEAT_REG = [270 522   523]; 
%FEAT_CLASS = [270 522   523];

FEAT = [270 522 523];

%CAT_FEAT = [3];
CAT_FEAT = [-1];
CAT_FEAT_EXP = [1 2 3 4 5 6 7 8 9 10 11]';
%%%%%%
# Xcont_reg = [];
# Xcat_reg = [];
# Xcont_class = [];
# Xcat_class = [];

# for k = 1:length(FEAT_REG)
#   if (sum( FEAT_REG(k) == CAT_FEAT ) >0)
#     ii = find( FEAT_REG(k) == CAT_FEAT );
#     cat = encodeCategoricalFeatures( [data(:, FEAT_REG(k)) ], index=-1,offset=-1, labels=CAT_FEAT_EXP(:,ii) );
#     Xcat_reg = [Xcat_reg cat];
#   else 
#     Xcont_reg = [Xcont_reg data(:, FEAT_REG(k) )];
#   endif
# endfor 

# for k = 1:length(FEAT_CLASS)
#   if (sum( FEAT_CLASS(k) == CAT_FEAT ) >0)
#     ii = find( FEAT_CLASS(k) == CAT_FEAT );     
#     cat = encodeCategoricalFeatures( [data(:, FEAT_CLASS(k)) ], index=-1,offset=-1, labels=CAT_FEAT_EXP(:,ii) );
#     Xcat_class = [Xcat_class cat];                                            
#   else                                                                        
#     Xcont_class = [Xcont_class data(:, FEAT_CLASS(k) )];                        
#   endif 
# endfor 


Xtrain = [];

for k = 1:length(FEAT)                                                                                                                                                                                
  Xtrain = [Xtrain data(:, FEAT(k) )];                                                                                                                                                       
endfor


%%%%%%
# Xtestcont_reg = [];
# Xtestcat_reg = [];
# Xtestcont_class = [];
# Xtestcat_class = [];

# for k = 1:length(FEAT_REG)
#   if (sum( FEAT_REG(k) == CAT_FEAT ) >0)
#     ii = find( FEAT_REG(k) == CAT_FEAT );
#     cat = encodeCategoricalFeatures( [data_test(:, FEAT_REG(k)) ], index=-1,offset=-1, labels=CAT_FEAT_EXP(:,ii) );
#     Xtestcat_reg = [Xtestcat_reg cat];
#   else 
#     Xtestcont_reg = [Xtestcont_reg data_test(:, FEAT_REG(k) )];
#   endif
# endfor 

# for k = 1:length(FEAT_CLASS)
#   if (sum( FEAT_CLASS(k) == CAT_FEAT ) >0)
#     ii = find( FEAT_CLASS(k) == CAT_FEAT );     
#     cat = encodeCategoricalFeatures( [data_test(:, FEAT_CLASS(k)) ], index=-1,offset=-1, labels=CAT_FEAT_EXP(:,ii) );
#     Xtestcat_class = [Xtestcat_class cat];                                            
#   else                                                                        
#     Xtestcont_class = [Xtestcont_class data_test(:, FEAT_CLASS(k) )];                        
#   endif 
# endfor 

Xtest = [];

for k = 1:length(FEAT)                                                                                                                                                                                
  Xtest = [Xtest data_test(:, FEAT(k) )];                                                                                                                                                       
endfor

%%%%%%%%%%%%%%%

X = [Xtrain; Xtest];

mtrain = size(Xtrain,1);

[Xnorm,mu,sigma] = featureNormalize(X);

printf("|-> feature scaling: mu ... \n");disp(mu);
printf("|-> feature scaling: sigma ... \n");disp(sigma);


Xtest_reg = Xnorm(mtrain+1:end,:);
Xtest_class = Xnorm(mtrain+1:end,:);

Xtest_reg = [ones(size(Xtest_reg,1),1) Xtest_reg];
Xtest_class = [ones(size(Xtest_class,1),1) Xtest_class];

X_reg = Xnorm(1:mtrain,:);
X_class = Xnorm(1:mtrain,:);

X_reg = [ones(size(X_reg,1),1) X_reg];
X_class = [ones(size(X_class,1),1) X_class];

y_loss = data(:,end);
y_def = (y_loss > 0) * 1 + (y_loss == 0)*0; 

y = [y_def y_loss];    

m = size(y,1);

[Xtrain_reg,ytrain,Xval_reg,yval] = splitTrainValidation(X_reg,y,0.90); 
[Xtrain_class,ytrain,Xval_class,yval] = splitTrainValidation(X_class,y,0.90);

ytrain_def = ytrain(:,1);
ytrain_loss = ytrain(:,2);

yval_def = yval(:,1);
yval_loss = yval(:,2);


%%%%%%%%%%%%%%%%%%%%%%%% DEFAULT  CLASSIFIER  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
printf("|--> DEFAULT CLASSIFIER  ...\n");

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
rand_ind_boot = randperm(size(Xtrain_boot,1));
Xtrain_boot = Xtrain_boot(rand_ind_boot,:);
ytrain_boot = ytrain_boot(rand_ind_boot,:);

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
  
ptrain = sigmoid(Xtrain_class * all_theta' );
pred_log_train = (ptrain > epsilon);

%%%%%%%%%%%%%%%%%%%%%%% LOSS REGRESSOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%%%%%%%%%%%%% BOOTSTRAP %%%%%%%%%
_Xval_reg = Xval_reg;
_yval_loss = yval_loss;
if (REGRESSOR_BOOTSTRAP)
  t = (pred_log_train == 1);
  tt = [];
  for rr = 1:size(Xtrain_reg,1)
    if ( t(rr) ) 
      tt = [tt; Xtrain_reg(rr,:)];
    endif 
  endfor 
  Xtrain_reg = tt;
  
  t = (pred_log_train == 1);
    tt = [];
    for rr = 1:size(ytrain_loss,1)
      if ( t(rr) ) 
        tt = [tt; ytrain_loss(rr,:)];
      endif 
    endfor 
  ytrain_loss = tt;
  
  t = (pred_log == 1);
    tt = [];
    for rr = 1:size(Xval_reg,1)
      if ( t(rr) ) 
        tt = [tt; Xval_reg(rr,:)];
      endif 
    endfor 
    _Xval_reg = tt;
    
  t = (pred_log == 1);
      tt = [];
      for rr = 1:size(yval_loss,1)
        if ( t(rr) ) 
          tt = [tt; yval_loss(rr,:)];
        endif 
      endfor 
  _yval_loss = tt;
endif 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%%% 3) FINDING BEST PARAMS LOSS REGRESSOR 
printf("|--> LOSS REGRESSOR  ...\n");
[p_opt,J_opt] = findOptP_RegLin(Xtrain_reg, ytrain_loss, _Xval_reg, _yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
[reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain_reg, ytrain_loss, _Xval_reg, _yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);
  
%% --> model parameters: p_opt , reg_lambda_opt
REGpars = [p_opt reg_lambda_opt];
printf("|--> OPTIMAL LINEAR REGRESSOR PARAMS -->  opt. number of polinomial degree (p_opt) = %i , opt. lambda = %f\n",p_opt,reg_lambda_opt);
%%dlmwrite ('REGpars.zat', REGpars);
  
%% --> performance
%%[Xtrain_poly,mu,sigma] = treatContFeatures(Xtrain_reg,p_opt);
Xtrain_poly = polyFeatures(Xtrain_reg,p_opt); 
rtheta = trainLinearReg(Xtrain_poly, ytrain_loss, reg_lambda_opt, 400);
%%[Xval_poly,mu,sigma] = treatContFeatures(Xval_reg,p_opt,1,mu,sigma);
  
_Xval_poly = polyFeatures(_Xval_reg,p_opt);
_pred_loss = predictLinearReg(_Xval_poly,rtheta);
_pred_loss = (_pred_loss < 0) .* 0 + (_pred_loss > 100) .* 100 +  (_pred_loss >= 0 & _pred_loss <= 100) .*  _pred_loss;
[_mae_reg] = MAE(_pred_loss, _yval_loss);
printf("|-> trained loss regressor. MAE on BOOTSRTAPPED cross validation set = %f  \n",_mae_reg);
  
Xval_poly = polyFeatures(Xval_reg,p_opt);
pred_loss = predictLinearReg(Xval_poly,rtheta);
pred_loss = (pred_loss < 0) .* 0 + (pred_loss > 100) .* 100 +  (pred_loss >= 0 & pred_loss <= 100) .*  pred_loss;
[mae_reg] = MAE(pred_loss, yval_loss);
printf("|-> trained loss regressor. MAE on REAL cross validation set = %f  \n",mae_reg);
  
%%%%%%%%%%%%% COMBINED PRED %%%%%%%%%
pred_comb_log = (pred_log == 0) .* 0 + (pred_log == 1) .* pred_loss;
[mae_log] = MAE(pred_comb_log, yval_loss);
printf("|-> COMBINED PREDICTION --> MAE on cross validation set = %f  (mae_reg    =%f) (F1    =%f) (ACC    =%f) \n", mae_log, mae_reg, F1, acc_log);
printf("|->        vs           --> bestMAE                     = %f  (bestMAE_REG=%f) (bestMAE_F1=%f) (bestMAE_ACC=%f) \n", bestMAE  , bestMAE_REG , bestMAE_F1, bestMAE_ACC );

if (mae_log < bestMAE | bestMAE < 0 )

%%% update 
bestMAE = mae_log;
bestMAE_P = p_opt;
bestMAE_RTHETA = rtheta;
bestMAE_Epsilon = epsilon;
bestMAE_ALLTHETA = all_theta;

bestMAE_REG = mae_reg;
bestMAE_F1 = F1;
bestMAE_ACC = acc_log;

bestMAE_init_perm = rand_indices;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf("|-> prediction on test set ...  \n");

%%%%%%%%%%% PREDICTION W/ THE BEST REGRESSOR & THE BEST CLASSIFIER 

####### loss prediction 
%%[Xtest_poly,mu,sigma] = treatContFeatures(Xtest_reg,bestP);
Xtest_poly = polyFeatures(Xtest_reg,p_opt); 
predtest_loss = predictLinearReg(Xtest_poly,rtheta);
predtest_loss = (predtest_loss < 0) .* 0 + (predtest_loss > 100) .* 100 +  (predtest_loss >= 0 & predtest_loss <= 100) .*  predtest_loss;

####### default prediction
ptval = sigmoid(Xtest_class * all_theta' );
predtest_log = (ptval > epsilon);

##### combinata 
predtest_comb = (predtest_log == 0) .* 0 + (predtest_log == 1) .* predtest_loss;

ids = data_test(:,1);
sub_comb = [ids predtest_comb];

dlmwrite ('sub_comb_log.csv', sub_comb,",");

  
endif 



endfor 




