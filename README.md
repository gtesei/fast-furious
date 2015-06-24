# fast-furious


## What is it?
  fast-furiuos gathers code (**R, Matlab/Octave, Python**), models and meta-models I need but I didn't found on the shelf.
  
## My model implementations 
  * **Regularized Neural Networks** (package ```neural``` **very fast 100% vectorized implementation of backpropagation** in Matlab/Octave)
    + for **basic use cases** just run command line ```>octave GO_Neural.m```
    + for **binary classification problems** use ```nnCostFunction``` cost function (multiclass still in beta) wrapped in ```trainNeuralNetwork```. *E.g. for fitting a neural neural network with 400 neurons at input layer, 25 neurons at hidden layer, 1 neuron (= binary classification) at output layer, 0.001 as regularization parameter, where trainset/testset has been already scaled and with the bias term added* 
    ```
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
    + for **tuning parameters** (number of neurons per layer, number of hidden layers, regularization parameter) by cross-validation use the ```findOptPAndHAndLambda``` function. *E.g. for finding the best number of neurons per layer (p_opt_acc), the best number of hidden layers (h_opt_acc), the best regularization parameter (lambda_opt_acc), using cross validation on a binary classification problem with accuracy as metric on a train set (80% of data) and cross validation set (20% of data) not scaled* 
    ```
    %% scale and add bias term 
    [train_data,mu,sigma] = treatContFeatures(train_data,1);
    [test_data,mu_val,sigma_val] = treatContFeatures(test_data,1,1,mu,sigma);
    
    %% split and randomize 
    [Xtrain,ytrain,Xval,yval] = splitTrainValidation(train_data,ytrain,0.80,shuffle=1);

    %% tuning parameters 
    [p_opt_acc,h_opt_acc,lambda_opt_acc,acc_opt,grid_j] = findOptPAndHAndLambda(Xtrain, ytrain, Xval, yval, ...
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
    + for **regression problems** use ```nnCostFunctionReg``` cost function 
    + for **large dataset** (e.g. **80GB train set on a machine with 8GB RAM**) use ```nnCostFunction_Buff``` that is a **buffered implementation of batch gradient descent**, i.e. it uses all train observations in each iteration vs. one observation as _stochastic gradient descent_ or k (k < number of observations on trainset) observations in each iteration as _mini-batch gradient descent_    
    + for **Neural Networks with EGS (= Extended Generalized Shuffle) interconnection pattern among layers** in regression problesm use ```nnCostFunctionRegEGS``` cost function 
    
  * **Regularized Linear and Polynomial Regression** (package ```linear_reg``` **very fast 100% vectorized implementation** in Matlab/Octave)
    + for **basic use cases** just run command line ```>octave GO_LinearReg.m```
    + for a **performance comparison** (=RMSE) among **(fast-furiuos) Regularized Polynomial Regression**, **(libsvm) epsilon-SVR**, **(libsvm) nu-SVR**, **(fast-furiuos) Neural Networks** on dataset *solubility* of [AppliedPredictiveModeling](http://appliedpredictivemodeling.com/) run command line ```>octave linear_reg/____testRegression.m```
  
