function [J grad] = nnCostFunction_Buff(nn_params, ...
                                   NNMeta, ...
                                   fX,ciX,ceX,fy,ciy,cey,_sep=',', b = 10000 ,lambda,featureScaled = 0)

m = countLines(fy,b); 
L = length(NNMeta.NNArchVect); 

Theta = cell(L-1,1);
Theta_grad = cell(L-1,1);
start = 1;
for i = 1:(L-1)
  Theta(i,1) = reshape(nn_params(start:start - 1 + NNMeta.NNArchVect(i+1) * (NNMeta.NNArchVect(i) + 1)), ...
                       NNMeta.NNArchVect(i+1), (NNMeta.NNArchVect(i) + 1));
  Theta_grad(i,1) = zeros(size(cell2mat(Theta(i,1))));
  start += NNMeta.NNArchVect(i+1) * (NNMeta.NNArchVect(i) + 1);
endfor 

X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
y = dlmread(fy,sep=_sep,[0,ciy,b-1,cey]);
_m = size(X,1);

% =========================================================================
%------------ FORWARD PROP
a = cell(L,1); 

if (featureScaled == 1) 
  a(1,1) = X;
else
  a(1,1) = [ones(_m,1) X];
endif 

z = cell(L,1);
for i = 2:L
 z(i,1) = cell2mat(a(i-1,1)) * cell2mat(Theta(i-1,1))';
 _a = sigmoid(cell2mat(z(i,1)));
 if (i < L)
  a(i,1) = [ones(_m,1) _a];
 else 
  a(i,1) = _a; 
 endif 
endfor 
 
yVect = zeros(m,NNMeta.NNArchVect(L));
for i = 1:NNMeta.NNArchVect(L)
  yVect(:,i) = (y == i);
endfor

hx = cell2mat(a(L,1));

S = ( -1 * innerProductMat_Y_1_X(yVect , log(hx)) -1 * innerProductMat_Y_1_X((1 .- yVect), log(1 .- hx)));
J = (1/m) * S;

%-------------BACK PROP
d = cell(L,1);

d(L,1) = cell2mat(a(L,1)) - yVect;
for i = fliplr(2:L-1)
 Theta_grad(i,1) = cell2mat(Theta_grad(i,1)) + cell2mat(d(i+1,1))' * cell2mat(a(i,1));
 d(i,1) = cell2mat(d(i+1,1)) * cell2mat(Theta(i,1));
 d(i,1) = cell2mat(d(i,1))(:,2:end);  
 d(i,1) = cell2mat(d(i,1)) .* sigmoidGradient(cell2mat(z(i,1))); 
endfor 
Theta_grad(1,1) = cell2mat(Theta_grad(1,1)) + cell2mat(d(2,1))' * cell2mat(a(1,1)); 

tai = cell(L-1,1);
for i = 1:(L-1)
  s = size(cell2mat(Theta(i,1)));
  tai(i,1) = [zeros(s(1),1),cell2mat(Theta(i,1))(:,2:end)];
endfor 

for i = 1:(L-1)
  Theta_grad(i,1) = (1/m) * (cell2mat(Theta_grad(i,1)) .+ (lambda)*cell2mat(tai(i,1)));
endfor 

%---------------------------- REGULARIZATION 
regTerm = 0; 

for i = 1:(L-1)
  tr = cell2mat(Theta(i,1))(:,2:end);
  tr = tr .* tr;
  regTerm = regTerm + (lambda/(2 * m)) * sum(tr(:));
endfor 

J = J + regTerm;

% =========================================================================

c = _m;
while ((_m == b) && (c < m) )
	X = dlmread(fX,sep=_sep,[c,ciX,c+b-1,ceX]);
	y = dlmread(fy,sep=_sep,[c,ciy,c+b-1,cey]);
  	_m = size(X,1)

	% =========================================================================
	%------------ FORWARD PROP
	a = cell(L,1); 
	
	if (featureScaled == 1) 
	  a(1,1) = X;
	else
	  a(1,1) = [ones(_m,1) X];
	endif 
	
	z = cell(L,1);
	for i = 2:L
	 z(i,1) = cell2mat(a(i-1,1)) * cell2mat(Theta(i-1,1))';
	 _a = sigmoid(cell2mat(z(i,1)));
	 if (i < L)
	  a(i,1) = [ones(_m,1) _a];
	 else 
	  a(i,1) = _a; 
	 endif 
	endfor 
	 
	yVect = zeros(m,NNMeta.NNArchVect(L));
	for i = 1:NNMeta.NNArchVect(L)
	  yVect(:,i) = (y == i);
	endfor
	
	hx = cell2mat(a(L,1));
	
	S = ( -1 * innerProductMat_Y_1_X(yVect , log(hx)) -1 * innerProductMat_Y_1_X((1 .- yVect), log(1 .- hx)));
	J += (1/m) * S;
	
	%-------------BACK PROP
	d = cell(L,1);
	
	d(L,1) = cell2mat(a(L,1)) - yVect;
	for i = fliplr(2:L-1)
	 Theta_grad(i,1) = cell2mat(Theta_grad(i,1)) + cell2mat(d(i+1,1))' * cell2mat(a(i,1));
	 d(i,1) = cell2mat(d(i+1,1)) * cell2mat(Theta(i,1));
	 d(i,1) = cell2mat(d(i,1))(:,2:end);  
	 d(i,1) = cell2mat(d(i,1)) .* sigmoidGradient(cell2mat(z(i,1))); 
	endfor 
	Theta_grad(1,1) = cell2mat(Theta_grad(1,1)) + cell2mat(d(2,1))' * cell2mat(a(1,1)); 
	
	%tai = cell(L-1,1);
	%for i = 1:(L-1)
	%  s = size(cell2mat(Theta(i,1)));
	%  tai(i,1) = [zeros(s(1),1),cell2mat(Theta(i,1))(:,2:end)];
	%endfor 
	
	for i = 1:(L-1)
	  Theta_grad(i,1) = (1/m) * (cell2mat(Theta_grad(i,1))) %%%%%%%%%%.+ (lambda)*cell2mat(tai(i,1)));
	endfor 
	
	% =========================================================================


	c += _m;
endwhile

% -------------------------------------------------------------

% Unroll gradients

grad = [];
for i = fliplr(1:L-1)
  grad =   [ cell2mat(Theta_grad(i))(:) ;  grad(:) ];
endfor

endfunction
