# fast-furious


## 1. What is it?
  fast-furiuos gathers code (**R, Matlab/Octave, Python**), models and meta-models I needed in my Machine Learning Lab but I didn't found on the shelf.
  
## 2. Requirements, installation and how to use fast-furious in your scripts 
fast-furious has been built in interpretable languages like R, Matlab/Octave, Python (hence, it does not require compilation) and **(Mac) OSX**, **Windows**, **Linux** are **fully supported**. 

### 2.1 Requirements
  * [Octave](http://www.gnu.org/software/octave/download.html) or Matlab is **mandatory** for fast-furious model implementations (*regularized neural networks, regularized linear and polynomial regression, regularized logistic regression*). If you are using only these fast-furious models Octave or Matlab installed on your machine is the only requirement. 
  * [R](http://www.r-project.org/) is **mandatory** for data process, feature engineering, model selection and model ensembling best practices 
  * [Python](https://www.python.org/downloads/) is **optional** as most of the python stuff is available in R as well 
  
### 2.2 Installation  
  Installation is pretty easy and quick. You can choose
  * to download the zip in the directory you like as **fast-furious base dir** and unzip  
  * or to use ```git``` in the directory you like as **fast-furious base dir** 
  
  ```
  git clone https://github.com/gtesei/fast-furious.git
  ```
  
### 2.3 How to use fast-furious in your Octave/Matlab scripts  
Assuming you are launching your Octave/Matlab script in fast-furious base dir, you just need to call at the begin of your script the fast-furious 
```menv``` function to set up the enviroment. Typically, your script should look like this 

```matlab
%% setting enviroment 
menv;

... here your stuff ...
```
For example, this is the code of fast-furious ```GO_Neural.m``` script located on fast-furious base dir: 
```matlab
%% setting enviroment 
menv;

%% load use cases and go  
source './neural/README.m';
go();
```

### 2.4 How to use fast-furious in your R scripts  
Assuming you are launching your R script in fast-furious base dir, you just need to ```source``` fast-furious resources at the begin of your script. For example, this is the code to perform imputation with fast-furious ```blackGuido``` function on a given data set _weather_ (excluding first two predictors) and using the best performing (RMSE) models among linear regression, KNN, PLS, Ridge regression, SVM, Cubist for continuous imputing predictors, and using the best performing (AUC) models among mode and SVM for categorical imputing predictors.   
```r
source("./data_process/Impute_Lib.R")

## imputing missing values ...
l = blackGuido (data = weather[,-c(1,2)], 
                RegModels = c("LinearReg","KNN_Reg", "PLS_Reg" , "Ridge_Reg" , "SVM_Reg", "Cubist_Reg")  , 
                ClassModels = c("Mode" , "SVMClass"), 
                verbose = T , 
                debug = F)
weather.imputed = l[[1]]
ImputePredictors = l[[2]]
DecisionMatrix = l[[3]]

weather.imputed = cbind(weather[,c(1,2)] , weather.imputed)
```


## 3. fast-furious model implementations 
### 3.1 Regularized Neural Networks 
Package ```neural``` **very fast 100% vectorized implementation of backpropagation** in Matlab/Octave.

 * for **basic use cases** just run command line (fast-furious base dir) 
 
    ```>octave GO_Neural.m```
    
 * for **binary classification problems** use ```nnCostFunction``` cost function (multiclass still in beta) wrapped in ```trainNeuralNetwork```. E.g. this is the code for fitting a neural neural network with 400 neurons at input layer, 25 neurons at hidden layer, 1 neuron (= binary classification) at output layer, 0.001 as regularization parameter, where trainset/testset has been already scaled and with the bias term added.
    ```matlab
    %% 400 neurons at input layer
    %% 25 neurons at hidden layer
    %% 1 neuron at output layer  
    NNMeta = buildNNMeta([400 25 1]); 
    
    %% regularization parameter 
    lambda = 0.001; 
    
    %% train on train set 
    [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 100, featureScaled = 1); 
    
    %% predict on train set 
    probs_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
    pred_train = (probs_train > 0.5);
    
    %% predict on test set 
    probs_test = NNPredictMulticlass(NNMeta, Theta , Xtest , featureScaled = 1);
    pred_test = (probs_test > 0.5);
    
    %% measure accuracy 
    acc_train = mean(double(pred_train == ytrain)) * 100;
    acc_test = mean(double(pred_test == ytest)) * 100;
    ```
 * for **tuning parameters on classification problems** (number of neurons per layer, number of hidden layers, regularization parameter) by cross-validation use the ```findOptPAndHAndLambda``` function. E.g. this is the code for finding the best number of neurons per layer (p_opt_acc), the best number of hidden layers (h_opt_acc), the best regularization parameter (lambda_opt_acc), using cross validation on a binary classification problem with accuracy as metric on a train set (80% of data) and cross validation set (20% of data) not scaled.
    ```matlab
    %% scale and add bias term 
    [train_data,mu,sigma] = treatContFeatures(train_data,1);
    [test_data,mu,sigma] = treatContFeatures(test_data,1,1,mu,sigma);
    
    %% split and randomize 
    [Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_data,ytrain,0.80,shuffle=1);

    %% tuning parameters 
    [p_opt_acc,h_opt_acc,lambda_opt_acc,acc_opt,tuning_grid] = findOptPAndHAndLambda(Xtrain, ytrain, Xval, yval, ...
  				featureScaled = 1 , 
					h_vec = [1 2 3 4 5 6 7 8 9 10] , ...
					lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10] , ...
					verbose = 1, doPlot=1 , ...
					iter = 200 , ...
					regression = 0 , num_labels = 1 );
                      
    %% train on full train set 
    NNMeta = buildNNMeta([(size(train_data,2)-1) (ones(h_opt_acc,1) .* p_opt_acc)' 1]');
    [Theta] = trainNeuralNetwork(NNMeta, train_data, ytrain, lambda_opt_acc , iter = 2000, featureScaled = 1);
  
    %% predict on train set 
    probs_train = NNPredictMulticlass(NNMeta, Theta , train_data , featureScaled = 1);
    pred_train = (probs_train > 0.5);
    acc_train = mean(double(pred_train == ytrain)) * 100;

    %% predict on test set 
    probs_test = NNPredictMulticlass(NNMeta, Theta , test_data , featureScaled = 1); 
    pred_test = (probs_test > 0.5);
    ```
 * for **regression problems** use ```nnCostFunctionReg``` cost function wrapped in ```trainNeuralNetworkReg```. E.g. this is the code for fitting a neural neural network with 400 neurons at input layer, 25 neurons at hidden layer, 1 neuron at output layer, 0.001 as regularization parameter, where trainset/testset has been already scaled and with the bias term added.
    ```matlab
    %% 400 neurons at input layer
    %% 25 neurons at hidden layer
    %% 1 neuron at output layer  
    NNMeta = buildNNMeta([400 25 1]); 
    
    %% regularization parameter 
    lambda = 0.001; 
    
    %% train on train set 
    [Theta] = trainNeuralNetworkReg(NNMeta, Xtrain, ytrain, lambda , iter = 200, featureScaled = 1);
    
    %% predict on train set 
    pred_train = NNPredictReg(NNMeta, Theta , Xtrain , featureScaled = 1);
    
    %% predict on test set 
    pred_test = NNPredictReg(NNMeta, Theta , Xtest , featureScaled = 1);
    
    %% measure RMSE 
    RMSE_train = sqrt(MSE(pred_train, ytrain));
    RMSE_test = sqrt(MSE(pred_test, ytest));
    ```
 * for **tuning parameters on regression problems** (number of neurons per layer, number of hidden layers, regularization parameter) by cross-validation use the ```findOptPAndHAndLambda``` function. E.g. this is the code for finding the best number of neurons per layer (p_opt_rmse), the best number of hidden layers (h_opt_rmse), the best regularization parameter (lambda_opt_rmse), using cross validation on a regression problem with RMSE as metric on a train set (80% of data) and cross validation set (20% of data) not scaled.
    ```matlab
    %% scale and add bias term 
    [train_data,mu,sigma] = treatContFeatures(train_data,1);
    [test_data,mu,sigma] = treatContFeatures(test_data,1,1,mu,sigma);
    
    %% split and randomize 
    [Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_data,ytrain,0.80,shuffle=1);

    %% tuning parameters 
    [p_opt_rmse,h_opt_rmse,lambda_opt_rmse,rmse_opt,tuning_grid] = findOptPAndHAndLambda(Xtrain, ytrain, Xval, yval, ...
    			featureScaled = 1 , 
					h_vec = [1 2 3 4 5 6 7 8 9 10] , ...
					lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10] , ...
					verbose = 1, doPlot=1 , ...
					iter = 200 , ...
					regression = 1 );
                      
    %% train on full train set 
    NNMeta = buildNNMeta([(size(train_data,2)-1) (ones(h_opt_rmse,1) .* p_opt_rmse)' 1]');
    [Theta] = trainNeuralNetworkReg(NNMeta, train_data, ytrain, lambda_opt_rmse , iter = 2000, featureScaled = 1);
  
    %% predict on train set 
    pred_train = NNPredictReg(NNMeta, Theta , Xtrain , featureScaled = 1);
    RMSE_train = sqrt(MSE(pred_train, ytrain));

    %% predict on test set 
    pred_test = NNPredictReg(NNMeta, Theta , Xtest , featureScaled = 1);
    RMSE_test = sqrt(MSE(pred_test, ytest));
    ```
 * for **large datasets** (e.g. **80GB train set on a machine with 8GB RAM**) use ```nnCostFunction_Buff``` (wrapped in ```trainNeuralNetwork_Buff```) that is a **buffered implementation of batch gradient descent**, i.e. it uses all train observations in each iteration vs. one observation as **stochastic gradient descent** or k (k < number of observations on trainset) observations in each iteration as **mini-batch gradient descent**. E.g. this is the code for for fitting a neural neural network with 400 neurons at input layer, 25 neurons at hidden layer, 1 neuron (= binary classification) at output layer, 0.001 as regularization parameter, from file  ```foXtrain ``` for predictors (columns from  ```ciX ``` to  ```ceX ```), and from file  ```fytrain ``` for labels (columns form  ```ciy ``` to  ```cey ```) and buffer equals to 10000 observations (= you load in memory 10000 observations each time).      
    ```matlab
    %% 400 neurons at input layer
    %% 25 neurons at hidden layer
    %% 1 neuron at output layer  
    NNMeta = buildNNMeta([400 25 1]); 
    
    %% regularization parameter 
    lambda = 0.001; 
  
    %% train (buffer = 10000 observations) 
    %% from file <foXtrain> (columns from <ciX> to <ceX>) as train data
    %% from file <fytrain> (columns form <ciy> to <cey>) as labels 
    [Theta_Buff] = trainNeuralNetwork_Buff(NNMeta,foXtrain,ciX,ceX, ... 
                            fytrain,ciy,cey, ... 
                            _sep=',',b=10000, ... 
                            lambda, iter = 50 , ... 
                            featureScaled = 0 , ... 
                            initialTheta = cell(0,0) );
    
    %% predict (buffer = 10000 observations) on train set 
    pred_val_bf = NNPredictMulticlass_Buff(NNMeta,foXval,ciX,ceX,Theta_Buff,10000,',',0);
    
    %% predict (buffer = 10000 observations) on test set 
    pred_train_bf = NNPredictMulticlass_Buff(NNMeta,foXtrain,ciX,ceX,Theta_Buff,10000,',',0);
    ```
 * for **Neural Networks with EGS (= Extended Generalized Shuffle) interconnection pattern among layers** in regression problesm use ```nnCostFunctionRegEGS``` cost function wrapped in ```trainNeuralNetworkRegEGS``` function. E.g. this is the code for fitting a neural neural network with 400 neurons at input layer, 25 neurons at hidden layer, 1 neuron (= binary classification) at output layer, 0.001 as regularization parameter, where trainset/testset has been already scaled and with the bias term added. 
    ```matlab
    %% 400 neurons at input layer
    %% 25 neurons at hidden layer
    %% 1 neuron at output layer  
    NNMeta = buildNNMeta([400 25 1]); 
    
    %% regularization parameter 
    lambda = 0.001; 
    
    %% train 
    [Theta] = trainNeuralNetworkRegEGS(NNMeta, Xtrain, ytrain, lambda , iter = 300, featureScaled = 1 );
    
    %% predict on train/test set 
    pred_train = NNPredictRegEGS(NNMeta, Theta , Xtrain , featureScaled = 1);
    pred_test = NNPredictRegEGS(NNMeta, Theta , Xtest , featureScaled = 1);
    
    %% measure MSE on train/test predictions 
    MSE_train = MSE(pred_train, ytrain);
    MSE_test = MSE(pred_test, ytest);
    ```
    
### 3.2 Regularized Linear and Polynomial Regression 
Package ```linear_reg``` **very fast 100% vectorized implementation** in Matlab/Octave
 * for **basic use cases** just run command line (fast-furious base dir) 
    
    ```>octave GO_LinearReg.m```
 * for a **performance comparison** (=RMSE) among **(fast-furiuos) Regularized Polynomial Regression**, **(libsvm) epsilon-SVR**, **(libsvm) nu-SVR**, **(fast-furiuos) Neural Networks** on dataset *solubility* of [AppliedPredictiveModeling](http://appliedpredictivemodeling.com/) run command line 
    
    ```>octave linear_reg/____testRegression.m```

 * for fitting a **linear regression** model use ```linearRegCostFunction``` wrapped in  ```trainLinearReg``` function. E.g. this is the code for fitting a regularized liner regression model with trainset/testset not scaled and with regularization parameter set to 0.001.   
    ```matlab
    %% feature scaling (trainset/testset) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p = 1);
    [Xtest,mu,sigma] = treatContFeatures(Xtest,p = 1,1,mu,sigma);
    
    %% regularization parameter 
    lambda = 0.001;
    
    %% train 
    [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
    
    %% predict
    pred_train =predictLinearReg(Xtrain,theta);
    pred_test = predictLinearReg(Xtest,theta);
    
    %% measure MSE
    mse_train = MSE(pred_train, ytrain);
    mse_test = MSE(pred_test, ytest);
    ```
 * for fitting a **linear regression** model using **the normal equation** instead of **batch gradient descent** use the ```normalEqn_RegLin``` function. **I recommend not to use the normal equation for large datasets**. E.g. this is the code for fitting a regularized liner regression model using **the normal equation** with trainset/testset not scaled and with regularization parameter set to 0.001. 
    ```matlab
    %% feature scaling (trainset/testset) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p = 1);
    [Xtest,mu,sigma] = treatContFeatures(Xtest,p = 1,1,mu,sigma);
    
    %% regularization parameter 
    lambda = 0.001;
    
    %% train 
    [theta] = normalEqn_RegLin(Xtrain,ytrain,lambda);
    
    %% predict 
    pred_train = predictLinearReg(Xtrain,theta);
    pred_test = predictLinearReg(Xtest,theta);
    
    %% measure performance 
    mse_train = MSE(pred_train, ytrain);
    mse_test = MSE(pred_test, ytest);
    ```
 * for fitting a **polynomial regression** model use ```linearRegCostFunction``` as well. Just set up the degree of the polynomial trasformation you like in the ```treatContFeatures``` function. E.g. this is the code for fitting a regularized liner regression model with trainset/testset not scaled and with regularization parameter set to 0.001 and **polynomial degree 5**.   
    ```matlab
    %% feature scaling (trainset/testset) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p = 5);
    [Xtest,mu,sigma] = treatContFeatures(Xtest,p = 5,1,mu,sigma);
    
    %% regularization parameter 
    lambda = 0.001;
    
    %% train 
    [theta] = trainLinearReg(Xtrain, ytrain, lambda , 200 );
    
    %% predict
    pred_train =predictLinearReg(Xtrain,theta);
    pred_test = predictLinearReg(Xtest,theta);
    
    %% measure MSE
    mse_train = MSE(pred_train, ytrain);
    mse_test = MSE(pred_test, ytest);
    ```
 * for **tuning parameters (on regression problems)** (degree of polynomial trasformation, regularization parameter) by cross-validation use the ```findOptPAndLambdaRegLin``` function. E.g. this is the code for finding the best degree of polynomial trasformation (p_opt_RMSE), the best regularization parameter (lambda_opt_RMSE), using cross validation on a regression problem with RMSE as metric on a train set and test set already scaled.
 
    ```matlab
    [p_opt_RMSE,lambda_opt_RMSE,RMSE_opt,grid]  = ... 
          findOptPAndLambdaRegLin(solTrainX, solTrainY, solTestX, solTestY, ...
            p_vec = [1 2 3 4 5 6 7 8 9 10 12 20]' , ...
            lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' , ...
            verbose = 1, initGrid = [] , initStart = -1 , iter=1000);
            
    printf(">>>>> found min RMSE=%f  with p=%i and lambda=%f \n", RMSE_opt , p_opt_RMSE , lambda_opt_RMSE );
    ```
 * for **large datasets** (e.g. **80GB train set on a machine with 8GB RAM**) you can use the ```trainLinearReg_MiniBatch``` function that is a **mini-batch gradient descent** implementation, i.e. it uses k observations (k < number of observations on trainset) in each iteration. E.g. this is the code for for fitting a linear regression model with 0.001 as regularization parameter, from file  ```foXtrain ``` for predictors (columns from  ```ciX ``` to  ```ceX ```), and from file  ```fytrain ``` for labels (columns form  ```ciy ``` to  ```cey ```) and buffer equals to 100 observations (= you load in memory 100 observations each time **and you use only these for complete a gradient descent iteration**).
 
    ```matlab
    %% regularization parameter 
    lambda = 0.001; 
  
    %% train (buffer = 100 observations) 
    %% from file <foXtrain> (columns from <ciX> to <ceX>) as train data
    %% from file <fytrain> (columns form <ciy> to <cey>) as labels 
    [theta_mb] = trainLinearReg_MiniBatch(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=100, _sep=',' , iter=200);
    
    %% predict 
    pred_train = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_mb,b=10000,_sep=',');
    pred_test = predictLinearReg_Buff(foXtest,ciX,ceX,theta_mb,b=10000,_sep=',');
    
    
    %% measure performance 
    mse_train = MSE(pred_train, ytrain);
    mse_test = MSE(pred_test, ytest);
    ```
 * for **large datasets** (e.g. **80GB train set on a machine with 8GB RAM**) you can use the ```trainLinearReg_Buff``` function that is a **buffered implementation of gradient descent**, i.e. it uses it uses all train observations in each iteration vs. one observation as **stochastic gradient descent** or k (k < number of observations on trainset) observations in each iteration as **mini-batch gradient descent**. E.g. this is the code for for fitting a linear regression model with 0.001 as regularization parameter, from file  ```foXtrain ``` for predictors (columns from  ```ciX ``` to  ```ceX ```), and from file  ```fytrain ``` for labels (columns form  ```ciy ``` to  ```cey ```) and buffer equals to 100 observations (= you load in memory 100 observations each time **but you use all train observations for complete a gradient descent iteration**).

    ```matlab
    %% regularization parameter 
    lambda = 0.001; 
  
    %% train (buffer = 100 observations) 
    %% from file <foXtrain> (columns from <ciX> to <ceX>) as train data
    %% from file <fytrain> (columns form <ciy> to <cey>) as labels 
    [theta_bf] = trainLinearReg_Buff(foXtrain,ciX,ceX,fytrain,ciy,cey,lambda, b=100, _sep=',' , iter=200);
    
    %% predict 
    pred_train = predictLinearReg_Buff(foXtrain,ciX,ceX,theta_bf,b=10000,_sep=',');
    pred_test = predictLinearReg_Buff(foXtest,ciX,ceX,theta_bf,b=10000,_sep=',');
    
    %% measure performance 
    mse_train = MSE(pred_train, ytrain);
    mse_test = MSE(pred_test, ytest);
    ```
### 3.3 Regularized Polynomial Logistic Regression 
Package ```logistic_reg``` **very fast 100% vectorized implementation** in Matlab/Octave

* for **basic use cases** just run command line (fast-furious base dir) 
    
    ```>octave GO_LogisticReg.m```
* for fitting a **logistic regression** model use ```lrCostFunction``` wrapped in  ```trainLogReg``` function. E.g. this is the code for fitting a regularized logistic regression model with trainset/testset not scaled and with regularization parameter set to 0.001. Note: in this code sketch insteaf of using 0.5 as probability threshold I use the ```selectThreshold``` that select the probability threshold maximizing F1-score.    
    ```matlab
    %% feature scaling (trainset/testset) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p = 1);
    [Xtest,mu,sigma] = treatContFeatures(Xtest,p = 1,1,mu,sigma);
    
    %% regularization parameter 
    lambda = 0.001;
    
    %% train 
    [theta] = trainLogReg(Xtrain, ytrain, lambda , iter = 200 );
    
    %% predict probabilities  
    probs_train = predictLogReg(Xtrain,theta);
    probs_test = predictLogReg(Xtest,theta);
	
    %% select threshold (instead of 0.5) on train data 
    %% Note: this usually should be done by cross-validation 
    thr = selectThreshold (ytrain,probs_train);
    
    %% predict labels   
   	pred_train = (probs_train > thr);
   	pred_train = (probs_test > thr);
    ```
* for fitting a **logistic polynomial regression** model use ```lrCostFunction``` as well. Just set up the degree of the polynomial trasformation you like in the ```treatContFeatures``` function. E.g. this is the code for fitting a regularized logistic regression model with trainset/testset not scaled, with regularization parameter set to 0.001 and **polynomial degree 10**.   
    ```matlab
    %% feature scaling (trainset/testset) 
    [Xtrain,mu,sigma] = treatContFeatures(Xtrain,p = 10);
    [Xtest,mu,sigma] = treatContFeatures(Xtest,p = 10,1,mu,sigma);
    
    %% regularization parameter 
    lambda = 0.001;
    
    %% train 
    [theta] = trainLogReg(Xtrain, ytrain, lambda , iter = 200 );
    
    %% predict probabilities  
    probs_train = predictLogReg(Xtrain,theta);
    probs_test = predictLogReg(Xtest,theta);
  
    %% select threshold (instead of 0.5) on train data 
    %% Note: this usually should be done by cross-validation 
    thr = selectThreshold (ytrain,probs_train);
    
    %% predict labels   
   	pred_train = (probs_train > thr);
   	pred_train = (probs_test > thr);
    ```
* for **tuning parameters (on classification problems)** (degree of polynomial trasformation, regularization parameter) by cross-validation use the ```findOptPAndLambdaRegLog``` function. E.g. this is the code for finding the best degree of polynomial trasformation, the best regularization parameter, using cross validation on a train set and cross-validation set already scaled. **Best parameters are found for metrics F1, precision, recall**. 
 
    ```matlab
    [p_opt_recall,lambda_opt_recall,p_opt_accuracy,lambda_opt_accuracy,p_opt_precision,lambda_opt_precision,p_opt_F1,lambda_opt_F1,grid] = ...
      findOptPAndLambdaRegLog(Xtrain, ytrain, Xval, yval)
      
    printf(">>>>> metric: F1        - found optimum with p=%i and lambda=%f \n", p_opt_F1 , lambda_opt_F1 );
    printf(">>>>> metric: precision - found optimum with p=%i and lambda=%f \n", p_opt_precision , lambda_opt_precision );
    printf(">>>>> metric: recall    - found optimum with p=%i and lambda=%f \n", p_opt_recall , lambda_opt_recall );
    ```
    
## References 
Most parts of fast-furious are based on the following resources: 
* Stanford professor Andrew NG resources: [1](http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning), [2](https://www.coursera.org/learn/machine-learning/home/info)
* J. Friedman, T. Hastie, R. Tibshirani, *The Elements of Statistical Learning*, Springer, 2009
* Max Kuhn and Kjell Johnson, *Applied Predictive Modeling*, Springer, 2013

Other resources: 
* G. James, D. Witten, T. Hastie, R. Tibshirani, *An Introduction to Statistical Learning*, Springer, 2013
* Hadley Wickham, *Advanced R*, Chapman & Hall/CRC The R Series, 2014 
* Paul S.P. Cowpertwait, Andrew V. Metcalfe, *Introductory Time Series with R*, Springer, 2009