function [Theta] = trainNeuralNetwork(NNMeta, X, y, lambda , iter = 200 , ... 
                        featureScaled = 0 , initialTheta = cell(0,0) , costWeigth = [1 1] )

%% ----- Initial params 
initial_nn_params = [];
L = length(NNMeta.NNArchVect); 
Theta = cell(L-1,1);
if (isempty(initialTheta))
  for i = 1:(L-1)
    Theta(i,1) = randInitializeWeights(NNMeta.NNArchVect(i),NNMeta.NNArchVect(i+1));
  endfor  
else 
  Theta = initialTheta;
endif 

for i = fliplr(1:L-1)
  initial_nn_params =  [ cell2mat(Theta(i))(:) ;  initial_nn_params(:) ];
endfor 

%% ----- Find minimum 
options = optimset('MaxIter', iter, 'GradObj', 'on');

costFunction = @(p) nnCostFunction(p, NNMeta, X, y, lambda, featureScaled , costWeigth);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


%% ----- Unroll params 
start = 1;
for i = 1:(L-1)
  Theta(i,1) = reshape(nn_params(start:start - 1 + NNMeta.NNArchVect(i+1) * (NNMeta.NNArchVect(i) + 1)), ...
                       NNMeta.NNArchVect(i+1), (NNMeta.NNArchVect(i) + 1));
  
  start += NNMeta.NNArchVect(i+1) * (NNMeta.NNArchVect(i) + 1);
endfor 


endfunction