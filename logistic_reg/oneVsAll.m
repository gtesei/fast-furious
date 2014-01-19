function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i
%   num_labels can be 1,2,3, ...

 m = size(X, 1);
 n = size(X, 2)-1;

 all_theta = zeros(num_labels, n + 1);

 % X = [ones(m, 1) X]; X it's passed that way already 

 for c = 1:num_labels

	initial_theta = zeros(n + 1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 60);
	[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
	all_theta(c,:) = theta;

 endfor

endfunction
