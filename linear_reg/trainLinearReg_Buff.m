function [theta] = trainLinearReg_Buff(fX,ciX,ceX,fy,ciy,cey,lambda, b=10000, _sep=',' , iter=200)

X = dlmread(fX,sep=_sep,[0,ciX,1,ceX]);
theta = zeros(size(X, 2), 1);

costFunction = @(t) linearRegCostFunction_Buff(fX,ciX,ceX,fy,ciy,cey,t,lambda,b=b,_sep=_sep);
options = optimset('MaxIter', iter, 'GradObj', 'on');
theta = fmincg(costFunction, theta, options);

endfunction