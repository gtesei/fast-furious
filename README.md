# fast-furious


## What is it?
  fast-furiuos gathers code (**R, Matlab/Octave, Python**), models and meta-models of my Kaggle competitions.
  
## My model implementations 
  * **Regularized Neural Networks** (package ```neural``` **very fast 100% vectorized implementation of backpropagation** in Matlab/Octave)
    + for basic use cases just run command line ```>octave GO_Neural.m```
    + for binary classification problems use _nnCostFunction.m_ cost function (multiclass still in beta) wrapped in _trainNeuralNetwork.m_. *E.g. for fitting a neural neural network with 400 neurons at input layer, 25 neurons at hidden layer, 1 neuron (= binary classification) at output layer, 0.001 as regularization parameter, where trainset/testset has been already scaled and with the bias term added* 
    ```
    %% 400 neurons at input layer
    %% 25 neurons at hidden layer
    %% 1 neuron at output layer  
    NNMeta = buildNNMeta([400 25 1]); 
    
    %% regularization parameter 
    lambda = 0.001; 
    
    [Theta] = trainNeuralNetwork(NNMeta, Xtrain, ytrain, lambda , iter = 100, featureScaled = 1); 
    pred_train = NNPredictMulticlass(NNMeta, Theta , Xtrain , featureScaled = 1);
    pred_val = NNPredictMulticlass(NNMeta, Theta , Xval , featureScaled = 1);
    
    acc_train = mean(double(pred_train == ytrain)) * 100;
    acc_val = mean(double(pred_val == yval)) * 100;
    ```
    + for regression problems use _nnCostFunctionReg.m_ cost function 
    + for large dataset (e.g. **80GB train set on a machine with 8GB RAM**) use _nnCostFunction_Buff.m_ that is a **buffered implementation of batch gradient descent**, i.e. it uses all train observations in each iteration vs. one observation as _stochastic gradient descent_ or k (k < number of observations on trainset) observations in each iteration as _mini-batch gradient descent_    
    + for **Neural Networks with EGS (= Extended Generalized Shuffle) interconnection pattern among layers** in regression problesm use _nnCostFunctionRegEGS.m_ cost function 
    
  * **Regularized Linear and Polynomial Regression** (package ```linear_reg``` in Matlab/Octave)
    + for basic use cases just run command line ```>octave GO_LinearReg.m```
    + for a performance comparison (=RMSE) among **(fast-furiuos) Regularized Polynomial Regression**, **(libsvm) epsilon-SVR**, **(libsvm) nu-SVR**, **(fast-furiuos) Neural Networks** on dataset *solubility* of [AppliedPredictiveModeling](http://appliedpredictivemodeling.com/) run command line ```>octave linear_reg/____testRegression.m```
  
## Some selected competitions  
  * **_8th/504_ - American Epilepsy Society Seizure Prediction Challenge** (package _competitions/seizure-prediction_)
  * _371st/3514_ - Otto Group Product Classification Challenge (package _competitions/otto-group-product-classification-challenge_)
  * _59th/485_ - Walmart Recruiting II: Sales in Stormy Weather (package _competitions/walmart-recruiting-sales-in-stormy-weather_)
