function [theta] = trainLinearReg_MiniBatch(fX,ciX,ceX,fy,ciy,cey,lambda, b=10000, _sep=',' , iter=200)

m = countLines(fy,b);

X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
y = dlmread(fy,sep=_sep,[0,ciy,b-1,cey]);
theta = zeros(size(X, 2), 1);

costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
options = optimset('MaxIter', iter, 'GradObj', 'on');
theta = fmincg(costFunction, theta, options);

_m = size(X,1);
c = _m;
while ( (_m == b) && (c < m) )
  X = dlmread(fX,sep=_sep,[c,ciX,c+b-1,ceX]);
  y = dlmread(fy,sep=_sep,[c,ciy,c+b-1,cey]);

  costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
  options = optimset('MaxIter', iter, 'GradObj', 'on');
  theta = fmincg(costFunction, theta, options);

  _m = size(X,1);
  c += _m;
endwhile

endfunction
