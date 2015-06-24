# Fast-Furious


## What is it?
  -----------
  Fast-Furiuos gathers code (**R, Matlab/Octave, Python**), models and meta-models of my Kaggle competitions.
  

## Highlights
  -----------
  * **_8th/504_ - American Epilepsy Society Seizure Prediction Challenge** (package _competitions/seizure-prediction_)
  * _371st/3514_ - Otto Group Product Classification Challenge (package _competitions/otto-group-product-classification-challenge_)
  * _59th/485_ - Walmart Recruiting II: Sales in Stormy Weather (package _competitions/walmart-recruiting-sales-in-stormy-weather_)
  
## My model implementations 
  -----------
  * **Neural Networks** (package _neural_)
    + for basic use cases just run *octave GO_Neural.m*
    + for binary classification problems use _nnCostFunction.m_ cost function (multiclass still in beta)
    + for regression problems use _nnCostFunctionReg.m_ cost function 
    + for large dataset (e.g. 80GB train set on a machine with 8GB RAM) use _nnCostFunction_Buff.m_ that is a **buffered implementation of batch gradient descent**, i.e. it uses all train observations in each iteration vs. one observation as _stochastic gradient descent_ or k (k < number of observations on trainset) observations in each iteration as _mini-batch gradient descent_    
    + for **Neural Networks with EGS (= Extended Generalized Shuffle) interconnection pattern among layers** in regression problesm use _nnCostFunctionRegEGS.m_ cost function 
    
    * **Regularized Linear and Polynomial Regression** (package *linear_reg*)
    + for basic use cases just run *octave GO_LinearReg.m*
  
