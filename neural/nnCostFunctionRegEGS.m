function [J grad] = nnCostFunctionRegEGS(nn_params, ...
                                   NNMeta, ...
                                   X, y, lambda, featureScaled = 0)
                                   
m = size(X, 1);
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



%------------ FORWARD PROP
a = cell(L,1); 

if (featureScaled == 1) 
  a(1,1) = X;
else
  a(1,1) = [ones(m,1) X];
endif

%-------- EGS
n = size( X ,2);
if (!featureScaled)
  n = n + 1;
end
links = ceil( n  ^ (1/(L-1)) );
%printf("link=%i L=%i\n",links,L);
%--------

z = cell(L,1);
for i = 2:L
 %printf("i=%i\n",i);
 _nn = size( cell2mat(Theta(i-1,1)) ,2);
 _mm = size( cell2mat(Theta(i-1,1)) ,1);
 Theta(i-1,1) = cell2mat(Theta(i-1,1)) .* EGS( _mm , _nn, links , (i-1)  );

 z(i,1) = cell2mat(a(i-1,1)) * cell2mat(Theta(i-1,1))';
_a = cell2mat(z(i,1));

%if (i == 2)
%  _a = sigmoid(cell2mat(z(i,1)));
%endif


 if (i < L)
  a(i,1) = [ones(m,1) _a];
 else 
  a(i,1) = _a; 
 endif

endfor 
 
yVect = y;
hx = cell2mat(a(L,1));

S = sum( (hx - yVect) .^ 2 );
J = 1/(2*m)* S;

%-------------BACK PROP
d = cell(L,1);

d(L,1) = cell2mat(a(L,1)) - yVect;

for i = fliplr(2:L-1)
 Theta_grad(i,1) = cell2mat(Theta_grad(i,1)) + cell2mat(d(i+1,1))' * cell2mat(a(i,1));

 _nn = size( cell2mat(Theta(i,1)) ,2);
 _mm = size( cell2mat(Theta(i,1)) ,1);
 Theta(i,1) = cell2mat(Theta(i,1)) .* EGS( _mm , _nn , links , i  );
 d(i,1) = cell2mat(d(i+1,1)) * cell2mat(Theta(i,1));

 d(i,1) = cell2mat(d(i,1))(:,2:end);

%if i == 2
% d(i,1) = cell2mat(d(i,1)) .* sigmoidGradient(cell2mat(z(i,1)));
%endif

endfor 
Theta_grad(1,1) = cell2mat(Theta_grad(1,1)) + cell2mat(d(2,1))' * cell2mat(a(1,1));

_nn = size( cell2mat(Theta_grad(1,1)) ,2);
_mm = size( cell2mat(Theta_grad(1,1)) ,1);
Theta_grad(1,1) = cell2mat(Theta_grad(1,1)) .* EGS( _mm , _nn , links , 1  );

tai = cell(L-1,1);
for i = 1:(L-1)
  s = size(cell2mat(Theta(i,1)));
  tai(i,1) = [zeros(s(1),1),cell2mat(Theta(i,1))(:,2:end)];
endfor 

for i = 1:(L-1)
  Theta_grad(i,1) = (1/m) * (cell2mat(Theta_grad(i,1)) .+ (lambda)*cell2mat(tai(i,1)));

  _nn = size( cell2mat(Theta_grad(i,1)) ,2);
  _mm = size( cell2mat(Theta_grad(i,1)) ,1);
  Theta_grad(i,1) = cell2mat(Theta_grad(i,1)) .* EGS( _mm , _nn , links , i  );

endfor

%---------------------------- REGULARIZATION 
regTerm = 0; 

for i = 1:(L-1)
  tr = cell2mat(Theta(i,1))(:,2:end);
  tr = tr .* tr;
  regTerm = regTerm + (lambda/(2 * m)) * sum(tr(:));
endfor 

J = J + regTerm;

% -------------------------------------------------------------

% Unroll gradients
grad = [];
for i = fliplr(1:L-1)
  grad =   [ cell2mat(Theta_grad(i))(:) ;  grad(:) ];
endfor

endfunction
