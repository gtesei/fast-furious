function [p] = NNPredictMulticlass(NNMeta, Theta , X , featureScaled = 0)

m = size(X, 1);
p = zeros(size(X, 1), 1);

L = length(NNMeta.NNArchVect);
h = cell(L-1,1);

for i = 1:L-1
  if (featureScaled & i == 1)
    h(i,1) = sigmoid(X * cell2mat(Theta(i,1))');
  elseif (i == 1)
    h(i,1) = sigmoid([ones(m, 1) X] * cell2mat(Theta(i,1))');
  else 
    h(i,1) = sigmoid([ones(m, 1) cell2mat(h(i-1,1))] * cell2mat(Theta(i,1))');
  endif
endfor 

[dummy, p] = max(cell2mat(h(L-1,1)), [], 2);

end
