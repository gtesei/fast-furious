
bestMAE_FEAT_REG = [270   522   523     3    30    55   100   103   142]; 
bestMAE_FEAT_CLASS = [270   522   523     2     3     9    19    68   135   142];

%% compute bias in Xtrain 
trainFile = "train_NO_NA_oct.zat";
data_train = dlmread([curr_dir "/dataset/loan_default/" trainFile]);

Xtcont_reg = [];
Xtcont_class = [];

for k = 1:length(bestMAE_FEAT_REG)
  Xtcont_reg = [Xtcont_reg data_train(:, bestMAE_FEAT_REG(k) )];
endfor

for k = 1:length(bestMAE_FEAT_CLASS)
  Xtcont_class = [Xtcont_class data_train(:, bestMAE_FEAT_CLASS(k) )];
endfor

[_Xtcont_reg,mu_reg,sigma_reg] = featureNormalize(Xtcont_reg);
[_Xtcont_class,mu_class,sigma_class] = featureNormalize(Xtcont_class);

%% compute bias in Xtest 
Xcont_reg = [];
Xcont_class = [];

for k = 1:length(bestMAE_FEAT_REG)
  Xcont_reg = [Xcont_reg data(:, bestMAE_FEAT_REG(k) )];
endfor

for k = 1:length(bestMAE_FEAT_CLASS)
  Xcont_class = [Xcont_class data(:, bestMAE_FEAT_CLASS(k) )];
endfor

[_Xcont_reg,mu_reg_test,sigma_reg_test] = featureNormalize(Xcont_reg);
[_Xcont_class,mu_class_test,sigma_class_test] = featureNormalize(Xcont_class);

printf("****** mu_reg\n");disp(mu_reg);
printf("****** sigma_reg\n");disp(sigma_reg);

printf("****** mu_class\n");disp(mu_class);
printf("****** sigma_class\n");disp(sigma_class);

printf("****** mu_reg_test\n");disp(mu_reg_test);
printf("****** sigma_reg_test\n");disp(sigma_reg_test);

printf("****** mu_class_test\n");disp(mu_class_test);
printf("****** sigma_class_test\n");disp(sigma_class_test);

dmu_reg = mu_reg_test - mu_reg;
dmu_class = mu_class_test - mu_class;

printf("****** dmu_reg\n");disp(dmu_reg);
printf("****** dmu_class\n");disp(dmu_class); 

sdmu_reg= sum(dmu_reg);
sdmu_class = sum(dmu_class);

printf("****** sdmu_reg\n");disp(sdmu_reg);
printf("****** sdmu_class\n");disp(sdmu_class);

                                                                                                                                                 
%%%%%%%%%%%%%% predict 
Xcont_reg = Xcont_reg - repmat(dmu_reg,size(Xcont_reg,1),1);
Xcont_class = Xcont_class - repmat(dmu_class,size(Xcont_class,1),1);

[Xcont_reg,mu_reg,sigma_reg] = featureNormalize(Xcont_reg,1,mu_reg,sigma_reg);
[Xcont_class,mu_class,sigma_class] = featureNormalize(Xcont_class,1,mu_class,sigma_class);

Xtest_reg = [Xcont_reg];
Xtest_reg = [ones(size(Xtest_reg,1),1) Xtest_reg];

Xtest_class = [Xcont_class];
Xtest_class = [ones(size(Xtest_class,1),1) Xtest_class];

Xtest_poly = polyFeatures(Xtest_reg,bestMAE_P);
predtest_loss = predictLinearReg(Xtest_poly,bestMAE_RTHETA);
predtest_loss = (predtest_loss < 0) .* 0 + (predtest_loss > 100) .* 100 +  (predtest_loss >= 0 & predtest_loss <= 100) .*  predtest_loss;

ptval = sigmoid(Xtest_class * bestMAE_ALLTHETA' );
predtest_log = (ptval > bestMAE_Epsilon);

predtest_comb = (predtest_log == 0) .* 0 + (predtest_log == 1) .* predtest_loss;

ids = data(:,1);
sub_comb = [ids predtest_comb];

dlmwrite ('sub_comb_log3.csv', sub_comb,",");
