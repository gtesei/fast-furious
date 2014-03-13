#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

%%%%%%%%%% SEGMENTED REGRESSION LOSS REGRESSOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%trainFile = "train_NO_NA_oct.zat";
trainFile = "train_NO_NA_oct_10K.zat"; 
%trainFile = "train_v2_NA_CI_oct.zat";

%testFile = "test_v2_NA_CI_oct.zat";   
testFile = "train_NO_NA_oct_10K.zat";  
%testFile = "test_impute_mean_oct.zat";


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

%all_theta = [];
%rtheta = [];

[m,n] = size(data);
rand_indices = randperm(m);
data = data(rand_indices,:);
%%data = data(1:10000,:);



%FEAT_REG = [270   522   523     3    30    55   100   103   142]; 
%FEAT_CLASS = [270   522   523     2     3     9    19    68   135   142];
%FEAT_REG = [270 522   523]; 
%FEAT_CLASS = [270 522   523];

%FEAT = [270 522 523 620];

FEAT = [270 522 523 403];

%CAT_FEAT = [3];
CAT_FEAT = [-1];
CAT_FEAT_EXP = [1 2 3 4 5 6 7 8 9 10 11]';

Xtrain = [];

for k = 1:length(FEAT)                                                                                                                                                                                
  Xtrain = [Xtrain data(:, FEAT(k) )];                                                                                                                                                       
endfor

Xtest = [];

for k = 1:length(FEAT)                                                                                                                                                                                
  Xtest = [Xtest data_test(:, FEAT(k) )];                                                                                                                                                       
endfor

%%%%%%%%%%%%%%%

X = [Xtrain; Xtest];


printf("|-> BEFORE NORMALIZING \n");
for (i = 1:4)
  _max = max(X(:,i));
  _min = min(X(:,i));
  printf("|-> max(%i): %f \n",i,_max);
  printf("|-> min(%i): %f \n",i,_min);
endfor

[Xnorm,mu,sigma] = featureNormalize(X);

printf("|-> AFTER NORMALIZING \n");
for (i = 1:4)
  _max = max(Xnorm(:,i));
  _min = min(Xnorm(:,i));
  printf("|-> max(%i): %f \n",i,_max);
  printf("|-> min(%i): %f \n",i,_min);
endfor

%%%% final 
X = [Xnorm(:,1) Xnorm(:,2) Xnorm(:,3) Xnorm(:,4)];

printf("|-> AFTER SELECTING \n");
for (i = 1:4)
  _max = max(X(:,i));
  _min = min(X(:,i));
  printf("|-> max(%i): %f \n",i,_max);
  printf("|-> min(%i): %f \n",i,_min);
endfor

mtrain = size(Xtrain,1);
Xtv = X(1:mtrain,:);
Xtest = X(mtrain+1:end,:);

Xtv = [ones(size(Xtv,1),1) Xtv];

y_loss = data(:,end);
y_def = (y_loss > 0) * 1 + (y_loss == 0)*0;
y = [y_def y_loss];    

%%%% constants 
th0 = 0.10;
ITER_MAX = 10;
srange = 0:0.1:0.8;
mu4 = mu(4);
sd4 = sigma(4);

%%%% out 
bestTH = -1;
bestF1 = -1;
bestACC = -1;

gths = zeros(length(srange),1);
gMAEs = zeros(length(srange),1);

n1s = zeros(length(srange),1);
n0s = zeros(length(srange),1);

for tt = 1:length(srange)
  t  = srange(tt);
  _th = th0+t;
  th = (_th-mu4)/sd4;
  
  MAEs = zeros(ITER_MAX,1);
  
  p = (X(:,4) > th);
  ptv = p(1:mtrain,:);
  ptest = p(mtrain+1:end,:);

  n1s(tt) = sum(ptv);
  n0s(tt) = length(ptv) - n1s(tt);

  [Xtv_s1,Xtv_s0] = splitBy(Xtv,ptv);
  [Xtest_s1,Xtest_s0] = splitBy(Xtest,ptest);
  [y_s1,y_s0] = splitBy(y,ptv);


  [Xtrain_s1,ytrain_s1,Xval_s1,yval_s1] = splitTrainValidation(Xtv_s1,y_s1,0.90); 
  [Xtrain_s0,ytrain_s0,Xval_s0,yval_s0] = splitTrainValidation(Xtv_s0,y_s0,0.90); 


  ytrain_def_s1 = ytrain_s1(:,1);
  ytrain_loss_s1 = ytrain_s1(:,2);
  yval_def_s1 = yval_s1(:,1);
  yval_loss_s1 = yval_s1(:,2); 

  ytrain_def_s0 = ytrain_s0(:,1); 
  ytrain_loss_s0 = ytrain_s0(:,2);
  yval_def_s0 = yval_s0(:,1);
  yval_loss_s0 = yval_s0(:,2); 
 
  for iter = 1:ITER_MAX
        
        printf("|----> threshold breakpoint:%f (scaled:%f) -- iter:%i/%i ...\n",_th,th,iter,ITER_MAX);

	
	%%%%%%%%%%%%%%%%%%%%%%%% REGRESSOR  S1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
	Xtrain = Xtrain_s1;
	ytrain = ytrain_s1;
	ytrain_loss = ytrain_loss_s1;
	ytrain_def = yval_def_s1;
	
	Xval = Xval_s1;
	yval = yval_s1;
	yval_def = yval_def_s1;
	yval_loss = yval_loss_s1; 
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	printf("|--> LOSS REGRESSOR   -- S1 --  ...\n");
	i = 95;
	%%%%%%%%%%%%% BOOTSTRAP %%%%%%%%%
	id0 = find(ytrain_loss == 0);
	id1 = find(ytrain_loss > 0);
	%printf("|--> 0/1 ratio:: %i / 100 .... found id0 - %i , id1- %i ... making equal length ...\n",i,length(id0),length(id1));
	r01 = floor( length(id1) * i / 100 ); 
	id0 = id0(1:r01,:);
	%printf("|--> made length(id0) == %i , length(id1) == %i ...\n",length(id0),length(id1));
	Xtrain_boot = [Xtrain(id1,:); Xtrain(id0,:)];
	ytrain_boot = [ytrain(id1,:); ytrain(id0,:)];
	
	%%% shuffle 
	rand_ind_boot = randperm(size(Xtrain_boot,1));
	Xtrain_boot = Xtrain_boot(rand_ind_boot,:);
	ytrain_boot = ytrain_boot(rand_ind_boot,:);
	
	ytrain_def_boot = ytrain_boot(:,1);
	ytrain_loss_boot = ytrain_boot(:,2);   
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%% 3) FINDING BEST PARAMS LOSS REGRESSOR 
	printf("|--> LOSS REGRESSOR trained only on 1' ...\n");
	[p_opt,J_opt] = findOptP_RegLin(Xtrain_boot, ytrain_loss_boot, Xval, yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
	[reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain_boot, ytrain_loss_boot, Xval, yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);
	  
	%% --> model parameters: p_opt , reg_lambda_opt
	REGpars = [p_opt reg_lambda_opt];
	printf("|--> OPTIMAL LINEAR REGRESSOR PARAMS -->  opt. number of polinomial degree (p_opt) = %i , opt. lambda = %f\n",p_opt,reg_lambda_opt);
	  
	%% --> performance
	Xtrain_poly = polyFeatures(Xtrain_boot,p_opt); 
	
	rtheta_in = [];
	%if ( isfield(rtheta_hash, num2str(p_opt)) )
	%  rtheta_in = getfield( rtheta_hash , num2str(p_opt) );
	%endif 
	
	rtheta = trainLinearReg(Xtrain_poly, ytrain_loss_boot, reg_lambda_opt, 400 , rtheta_in);
	
	%rtheta_hash = setfield(rtheta_hash , num2str(p_opt) , rtheta );
	  
	Xval_poly = polyFeatures(Xval,p_opt);
	pred_loss1 = predictLinearReg(Xval_poly,rtheta);
	pred_loss1 = (pred_loss1 < 0) .* 0 + (pred_loss1 > 100) .* 100 +  (pred_loss1 >= 0 & pred_loss1 <= 100) .*  pred_loss1;
	[mae_reg] = MAE(pred_loss1, yval_loss);  
        printf("|-> trained loss regressor S1. MAE on cross validation set = %f  \n",mae_reg);
	

        
        %%%%%%%%%%%%%%%%%%%%%%%% LOSS REGRESSOR  S0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
	Xtrain = Xtrain_s0;
	ytrain = ytrain_s0;
	ytrain_loss = ytrain_loss_s0;
	ytrain_def = yval_def_s0;

	Xval = Xval_s0;
	yval = yval_s0;
	yval_def = yval_def_s0;
	yval_loss = yval_loss_s0; 
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	printf("|--> DEFAULT CLASSIFIER   -- S0 --  ...\n");
	i = 95;
	%%%%%%%%%%%%% BOOTSTRAP %%%%%%%%%
	id0 = find(ytrain_loss == 0);
	id1 = find(ytrain_loss > 0);
	%printf("|--> 0/1 ratio:: %i / 100 .... found id0 - %i , id1- %i ... making equal length ...\n",i,length(id0),length(id1));
	r01 = floor( length(id1) * i / 100 ); 
	id0 = id0(1:r01,:);
	%printf("|--> made length(id0) == %i , length(id1) == %i ...\n",length(id0),length(id1));
	Xtrain_boot = [Xtrain(id1,:); Xtrain(id0,:)];
	ytrain_boot = [ytrain(id1,:); ytrain(id0,:)];

	%%% shuffle 
	rand_ind_boot = randperm(size(Xtrain_boot,1));
	Xtrain_boot = Xtrain_boot(rand_ind_boot,:);
	ytrain_boot = ytrain_boot(rand_ind_boot,:);

	ytrain_def_boot = ytrain_boot(:,1);
	ytrain_loss_boot = ytrain_boot(:,2);   
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%% 3) FINDING BEST PARAMS LOSS REGRESSOR 
	printf("|--> LOSS REGRESSOR trained only on 1' ...\n");
	[p_opt,J_opt] = findOptP_RegLin(Xtrain_boot, ytrain_loss_boot, Xval, yval_loss, p_vec = [1 2 3 4 5 6 7 8 9 10 12 20 60]' , lambda=0);
	[reg_lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain_boot, ytrain_loss_boot, Xval, yval_loss, lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , p=p_opt);

	%% --> model parameters: p_opt , reg_lambda_opt
	REGpars = [p_opt reg_lambda_opt];
	printf("|--> OPTIMAL LINEAR REGRESSOR PARAMS -->  opt. number of polinomial degree (p_opt) = %i , opt. lambda = %f\n",p_opt,reg_lambda_opt);

	%% --> performance
	Xtrain_poly = polyFeatures(Xtrain_boot,p_opt); 

	rtheta_in = [];
	%if ( isfield(rtheta_hash, num2str(p_opt)) )
	%  rtheta_in = getfield( rtheta_hash , num2str(p_opt) );
	%endif 

	rtheta = trainLinearReg(Xtrain_poly, ytrain_loss_boot, reg_lambda_opt, 400 , rtheta_in);

	%rtheta_hash = setfield(rtheta_hash , num2str(p_opt) , rtheta );

	Xval_poly = polyFeatures(Xval,p_opt);
	pred_loss0 = predictLinearReg(Xval_poly,rtheta);
	pred_loss0 = (pred_loss0 < 0) .* 0 + (pred_loss0 > 100) .* 100 +  (pred_loss0 >= 0 & pred_loss0 <= 100) .*  pred_loss0;
	[mae_reg] = MAE(pred_loss0, yval_loss);  
	printf("|-> trained loss regressor S0. MAE on cross validation set = %f  \n",mae_reg);


        
        %%%%% --> merge 
        pred_loss01 = [pred_loss1;pred_loss0];
        yval_loss = [yval_loss_s1;yval_loss_s0];
        [mae_reg01] = MAE(pred_loss01, yval_loss);  
        printf("|-> trained loss regressor GLOBAL. MAE on cross validation set = %f  \n",mae_reg01);
        
        MAEs(iter) = mae_reg01;
        

  endfor 
  
  MAE_mean = mean(MAEs);
  
  gths(tt) = th;
  gMAE(tt) = MAE_mean;
  
  fprintf("\n Regressor GLOBAL - END OF THERESOLD %f    MAE=%f \n", th, MAE_mean);
  
endfor 

  [bestMAE , idx] = max(gMAEs);
  bestTH = gths(idx);
  
  
  fprintf("\n Regressor GLOBAL - best th=%f  MAE=%f \n", bestTH, bestMAE);
  
  %%plot 
  plot(srange, gMAEs);
  %text(p_opt+1,J_opt+6,"Optimal Polynomial Degree","fontsize",10);
  title(sprintf('Segmented Regression'));
  xlabel('Breakpoint X4')
  ylabel('MAE')
  max_X = max(srange);
  max_Y = max(gMAEs);
  axis([0 max_X 0 max_Y]);
  %%legend('Train', 'Cross Validation')


