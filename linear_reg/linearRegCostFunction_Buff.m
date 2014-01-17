function [J, grad] = linearRegCostFunction_Buff(fX,ciX,ceX,fy,ciy,cey, theta, lambda,b=10000, _sep=',')

m = countLines(fy,b); 
J = 0; % cost
grad = zeros(size(theta)); % gradient 

X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
y = dlmread(fy,sep=_sep,[0,ciy,b-1,cey]);
_m = size(X,1);

% ====================== 
tai = [0;theta(2:end,1)];
J = 1/(2*m) * sum((X * theta - y) .^ 2) + (lambda/(2*m)) * (tai' * tai);
grad =   (1/m)*(X' * (X * theta -y)) + (lambda/m)*tai;
% =========================================================================

c = _m;
while ((_m == b) && (c < m) )
       
    X = dlmread(fX,sep=_sep,[c,ciX,c+b-1,ceX]);
    y = dlmread(fy,sep=_sep,[c,ciy,c+b-1,cey]);
    _m = size(X,1);
       
	% ====================== 
    #tai = [0;theta(2:end,1)];
	J += 1/(2*m) * sum((X * theta - y) .^ 2);
	grad +=   (1/m)*(X' * (X * theta -y));
	% =========================================================================
	
  	c += _m;
endwhile

grad = grad(:);

endfunction 
