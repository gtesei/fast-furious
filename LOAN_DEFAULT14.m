#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

%%%%%%%%%% SEGMENTED REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% best th(scaled)=-0.125468 th(not scaled)=0.5 ,  F1=0.745897 ACCURACY=93.749759
%%%% n1s = 24997 , n0s = 26943
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

trainFile = "train_NO_NA_oct.zat";
%trainFile = "train_NO_NA_oct_10K.zat"; 
%trainFile = "train_v2_NA_CI_oct.zat";

%testFile = "test_v2_NA_CI_oct.zat";   
%testFile = "train_NO_NA_oct_10K.zat";  
testFile = "test_impute_mean_oct.zat";


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
gF1s = zeros(length(srange),1);
gACCs = zeros(length(srange),1);
n1s = zeros(length(srange),1);
n0s = zeros(length(srange),1);

for tt = 1:length(srange)
  t  = srange(tt);
  _th = th0+t;
  th = (_th-mu4)/sd4;
  
  F1s = zeros(ITER_MAX,1);
  ACCs = zeros(ITER_MAX,1);
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

	
	%%%%%%%%%%%%%%%%%%%%%%%% DEFAULT  CLASSIFIER  S1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
	Xtrain = Xtrain_s1;
	ytrain = ytrain_s1;
	ytrain_loss = ytrain_loss_s1;
	ytrain_def = yval_def_s1;
	
	Xval = Xval_s1;
	yval = yval_s1;
	yval_def = yval_def_s1;
	yval_loss = yval_loss_s1; 
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	printf("|--> DEFAULT CLASSIFIER   -- S1 --  ...\n");
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
	
	lambda_log = 0;
	[all_theta] = oneVsAll(Xtrain_boot, ytrain_def_boot, 1, lambda_log ,800);
	pval = sigmoid(Xval * all_theta' );
	[epsilon F1] = selectThreshold(yval_def, pval);
	%fprintf("\n found bestEpsilon: %f       F1:%f      \n",epsilon,F1);
	pred_log = (pval > epsilon);
	acc_log = mean(double(pred_log == yval_def)) * 100;
	fprintf("\n Logistic classifier S1 - training set accuracy (p=%i,lambda=%f): %f\n", 1,lambda_log,acc_log);
	  
	ptrain = sigmoid(Xtrain * all_theta' );
        pred_log_train = (ptrain > epsilon);
        
        %% ---> out 
        pred_logs1 = pred_log;
        
        %%%%%%%%%%%%%%%%%%%%%%%% DEFAULT  CLASSIFIER  S0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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

	lambda_log = 0;
	[all_theta] = oneVsAll(Xtrain_boot, ytrain_def_boot, 1, lambda_log ,800);
	pval = sigmoid(Xval * all_theta' );
	[epsilon F1] = selectThreshold(yval_def, pval);
	%fprintf("\n found bestEpsilon: %f       F1:%f      \n",epsilon,F1);
	pred_log = (pval > epsilon);
	acc_log = mean(double(pred_log == yval_def)) * 100;
	fprintf("\n Logistic classifier S0 - training set accuracy (p=%i,lambda=%f): %f\n", 1,lambda_log,acc_log);

	ptrain = sigmoid(Xtrain * all_theta' );
        pred_log_train = (ptrain > epsilon);
        
        %% ---> out 
        pred_logs0 = pred_log;
        
        %%%%% --> merge 
        pred_log01 = [pred_logs1;pred_logs0];
        yval_def = [yval_def_s1;yval_def_s0];
        [epsilon F1] = selectThreshold(yval_def, pred_log01);
        acc_log = mean(double(pred_log01 == yval_def)) * 100;
        
        fprintf("\n Logistic classifier GLOBAL - F1=%f ACCURACY=%f\n", F1, acc_log);
        
        F1s(iter) = F1;
        ACCs(iter) = acc_log;

  endfor 
  
  F1_mean = mean(F1s);
  ACC_mean = mean(ACCs);
  
  gths(tt) = th;
  gF1s(tt) = F1_mean;
  gACCs(tt) = ACC_mean;
  
  fprintf("\n Logistic classifier GLOBAL - END OF THERESOLD %f    F1=%f ACCURACY=%f\n", th, F1_mean, ACC_mean);
  
endfor 

  [bestF1 , idx] = max(gF1s);
  bestTH = gths(idx);
  bestACC = gACCs(idx);
  
  fprintf("\n Logistic classifier GLOBAL - best th=%f  F1=%f ACCURACY=%f\n", bestTH, bestF1, bestACC);
  
  %%plot 
  plot(srange, gF1s);
  %text(p_opt+1,J_opt+6,"Optimal Polynomial Degree","fontsize",10);
  title(sprintf('Segmented Regression'));
  xlabel('Breakpoint X4')
  ylabel('F1-score')
  max_X = max(srange);
  max_Y = max(gF1s);
  axis([0 max_X 0 max_Y]);
  %%legend('Train', 'Cross Validation')


