#! /opt/local/bin/octave -qf 

%%%% setting enviroment 
menv;

trainFile = "train_nn.csv"; 
testFile = "test_nn.csv"; 

printf("|--> loading Xtrain, ytrain files ...\n");
train = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/" trainFile]); 
X_test = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/" testFile]); 

train = train(2:end,:); ## elimina le intestazioni del csv 
X_test = X_test(2:end,:);

y = train(:,end);
target = train(:,1); ## la prima colonna e' target ...
X = train(:,2:(end-1));

clear train;

### feature scaling and cenering ...
[X,mu,sigma] = treatContFeatures(X,1); 
[X_test,mu,sigma] = treatContFeatures(X_test,1,1,mu,sigma);

### cv ...
perc_train = 0.8;
[m,n] = size(X);
rand_indices = randperm(m);
[Xtrain,ytrain,Xval,yval] = splitTrainValidation(X(rand_indices,:),y(rand_indices,:),perc_train);

## traget
mtrain = floor(m * perc_train);
target_train = target(1:mtrain);
target_xval = target(mtrain+1:end);

###### train mediator ...
#id0 = find(ytrain == 0);
#id1 = find(ytrain == 1);
#mean0 = mean(Xtrain(id0,2:end));
#mean1 = mean(Xtrain(id1,2:end));
#meanT = mean(Xtrain);
#mean01 = (mean0+mean1)/2;
#Mean1InfMean0 = mean1 < mean0;
#t = (Xval(:,2:end) < mean01 ) .* Mean1InfMean0;
#pred_val = mode(t')';
#printClassMetrics(pred_val,yval);
                
#---> con 61 variabili tutte le previsioni a 0




###### train NN ...
num_label = 1;
NNMeta = buildNNMeta([(n-1) (n-1) num_label]);disp(NNMeta);

ww = [1 1];
[Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, 0 , iter = 450, featureScaled = 1 , initialTheta = cell(0,0) , costWeigth = ww);

_pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
thr = selectThreshold (ytrain,_pred_train);
pred_train = (_pred_train > thr);
printClassMetrics(pred_train,ytrain);
gini_train = NormalizedWeightedGini(target_train,Xtrain(:,2),_pred_train);
printf("NormalizedWeightedGini = %f \n", gini_train );

_pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
thr_xval = selectThreshold (yval,_pred_val);
pred_val = (_pred_val > thr);
printClassMetrics(pred_val,yval);
gini_xval = NormalizedWeightedGini(target_xval,Xval(:,2),_pred_val);
printf("NormalizedWeightedGini = %f \n", gini_xval );


######## NN tuning ....
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];
weigh_min = 1; weigth_max = 450;
weigth_sp = floor((weigth_max-weigh_min)/15);
grid = zeros(16 * length(lambda_vec),7);

acc_val_best = 0;
lambda_best = 0;
w_best = 1;

gi = 0;
for lambdaIdx = 1:length(lambda_vec)
    for w = weigh_min:weigth_sp:weigth_max

        gi = gi + 1;
        grid(gi,1) = lambda_vec(lambdaIdx);  ### grid - 1: lambda
        grid(gi,2) = w;   ### grid - 2: weigth

        printf("lambda = %f , weigth = %i \n", lambda_vec(lambdaIdx) , w );

        ww = [w 1];

        [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda_vec(lambdaIdx) , iter = 450, featureScaled = 1 , initialTheta = cell(0,0) , costWeigth = ww);

        _pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
        thr = selectThreshold (ytrain,_pred_train);
        pred_train = (_pred_train > thr);

        _pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
        pred_val = (_pred_val > thr);

        acc_train = mean(double(pred_train == ytrain)) * 100;
        acc_val = mean(double(pred_val == yval)) * 100;

        gini_train = NormalizedWeightedGini(target_train,Xtrain(:,2),_pred_train);
        gini_xval = NormalizedWeightedGini(target_xval,Xval(:,2),_pred_val);

        printf("|--> acc_val == %f , acc_train == %f  , gini_train = %f , gini_xval = %f \n",acc_val,acc_train,gini_train,gini_xval);
        printf("|--> *** TRAIN STATS ***\n");
        printClassMetrics(pred_train,ytrain);
        printf("|--> *** XVAL STATS ***\n");
        printClassMetrics(pred_val,yval);

        if (acc_val_best < acc_val)
            acc_val_best = acc_val;
            lambda_best =lambda_vec(lambdaIdx);
            w_best = w;
        endif

        grid(gi,3) = acc_val;        ### grid - 3: acc_val
        grid(gi,4) = acc_train;      ### grid - 4: acc_train
        grid(gi,5) = thr;            ### grid - 4: thr
        grid(gi,6) = gini_xval;      ### grid - 4: gini_xval
        grid(gi,7) = gini_train;     ### grid - 4: gini_train
    endfor
endfor

printf("|--> best accuracy = %f , lambda_best = %f ,  w_best = %f \n",acc_val_best,lambda_best,w_best);
dlmwrite ('grid_v2.zat', grid);


grid_sort = sortrows(grid,-3);
grid_sel = [0.3 145; 0.3 325; 0.01 325; 0.1 433; 0.001 61];
for i = 1:length(grid_sel)
  disp(grid_sel(i,:));
  lambda = grid_sel(i,1);
   w = grid_sel(i,2);
   ww = [w 1];
   [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 450, featureScaled = 1 , initialTheta = cell(0,0) , costWeigth = ww);

   thr = selectThreshold (ytrain,_pred_train);
   pred_train = (_pred_train > thr);

   _pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
   thr = selectThreshold (yval,_pred_val);
   pred_val = (_pred_val > thr);

    acc_train = mean(double(pred_train == ytrain)) * 100;
    acc_val = mean(double(pred_val == yval)) * 100;

   printf("|--> acc_val == %f vs acc_train == %f \n",acc_val,acc_train);

end

#### lancio con lambda = 0 e w = 1 



######## NN predicting on train ....
##params
w_opt = 1;
lambda_best = 0;
NNMeta = buildNNMeta([(n-1) (n-1) num_label]);disp(NNMeta);

## resempling full data set 
id0 = find(y == 0);
id1 = find(y == 1);
n1 = floor ((sum(y == 1) / size(y,1))^(-1));
Xtrain1 = X(id1,:);
Xtrain0 = X(id0,:);
Xtrain1i = Xtrain1;
for i = 1:n1
    Xtrain1 = [ Xtrain1i ; Xtrain1 ];
endfor

y0 = zeros(size(Xtrain0,1),1);
y1 = ones(size(Xtrain1,1),1);
y = [y0 ; y1];
X = [Xtrain0 ; Xtrain1];


## training and predicting
[Theta] = trainNeuralNetwork(NNMeta, X, y, lambda_best , iter = 600, featureScaled = 1 , initialTheta = cell(0,0) , costWeigth = [w_opt 1]);
_pred_train = NNPredictMulticlass(NNMeta, Theta , X , featureScaled = 1);
_pred_test = NNPredictMulticlass(NNMeta, Theta , X_test , featureScaled = 1);
thr = selectThreshold (y,_pred_train);
printf("|--> thr == %f  \n",thr);
pred_train = (_pred_train > thr);
pred_test = (_pred_test > thr);
printClassMetrics(pred_train,y);
dlmwrite ('pred_nn_test.zat', _pred_test);
dlmwrite ('pred_nn_test01.zat', pred_test);


######## resampling
id0 = find(ytrain == 0);
id1 = find(ytrain == 1);
n1 = floor ((sum(ytrain == 1) / size(ytrain,1))^(-1));

Xtrain1 = Xtrain(id1,:);
Xtrain0 = Xtrain(id0,:);
Xtrain1i = Xtrain1;
for i = 1:n1
    Xtrain1 = [ Xtrain1i ; Xtrain1 ];
endfor

ytrain0 = zeros(size(Xtrain0,1),1);
ytrain1 = ones(size(Xtrain1,1),1);

ytrain = [ytrain0 ; ytrain1];
Xtrain = [Xtrain0 ; Xtrain1];

[m,n] = size(Xtrain);
rand_indices = randperm(m);
ytrain = ytrain(rand_indices,:);
Xtrain = Xtrain(rand_indices,:);




