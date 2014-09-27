function [all_theta] = trainLogReg(X, y, lambda,iter=60, initialTheta=[])

theta = zeros(size(X, 2), 1);
if ( size(initialTheta,1) != 0 )
  theta = _theta;
endif

options = optimset('GradObj', 'on', 'MaxIter', iter);
[theta] = fmincg (@(t)(lrCostFunction(t, X, y, lambda)), initial_theta, options);


endfunction
