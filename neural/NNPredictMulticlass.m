function [p] = NNPredictMulticlass(NNMeta, Theta , X)

m = size(X, 1);
p = zeros(size(X, 1), 1);

L = length(NNMeta.NNArchVect);
h = cell(L-1,1);
h(1,1) = sigmoid([ones(m, 1) X] * cell2mat(Theta(1,1))');
for i = 2:L-1
  h(i,1) = sigmoid([ones(m, 1) cell2mat(h(i-1,1))] * cell2mat(Theta(i,1))');
endfor 

%h1 = sigmoid([ones(m, 1) X] * Theta1');
%h2 = sigmoid([ones(m, 1) h1] * Theta2');
%[dummy, p] = max(h2, [], 2);


[dummy, p] = max(cell2mat(h(L-1,1)), [], 2);

end
