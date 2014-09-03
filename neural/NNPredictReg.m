function [pred] = NNPredictReg(NNMeta, Theta , X , featureScaled = 0)

m = size(X, 1);

L = length(NNMeta.NNArchVect);
h = cell(L-1,1);
for i = 1:L-1
  if (featureScaled & i == 1)
    h(i,1) = X * cell2mat(Theta(i,1))';
  elseif (i == 1)
    h(i,1) = [ones(m, 1) X] * cell2mat(Theta(i,1))';
  else 
    h(i,1) = [ones(m, 1) cell2mat(h(i-1,1))] * cell2mat(Theta(i,1))';
  endif
endfor 

pred = cell2mat(h(L-1,1));
endfunction 
