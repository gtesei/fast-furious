function [theta] = trainLinearRegLiberty(X, y, lambda , iter = 200 , _theta=[])
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.

% Initialize Theta
theta = zeros(size(X, 2), 1);
if ( size(_theta,1) != 0 )
theta = _theta;
endif

% Create "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunctionLiberty(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', iter, 'GradObj', 'on');

% Minimize using fmincg
theta = fmincg(costFunction, theta, options);

endfunction
