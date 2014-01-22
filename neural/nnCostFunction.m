function [J grad] = nnCostFunction(nn_params, ...
                                   NNMeta, ...
                                   X, y, lambda)
                                   
m = size(X, 1);
L = length(NNMeta.NNArchVect); 

Theta = cell(L-1,1);
Theta_grad = cell(L-1,1);
start = 1;
for i = 1:(L-1)
  Theta(i,1) = reshape(nn_params(start:start - 1 + NNMeta.NNArchVect(i+1) * (NNMeta.NNArchVect(i) + 1)), ...
                       NNMeta.NNArchVect(i+1), (NNMeta.NNArchVect(i) + 1));
  Theta_grad(i,1) = zeros(size(Theta(i,1)));  
  start += NNMeta.NNArchVect(i+1) * (NNMeta.NNArchVect(i) + 1);
endfor 

%Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));
%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer_size + 1));

%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

%------------ FORWARD PROP
a = cell(L,1); a(1,1) = [ones(m,1) X];
z = cell(L,1);
for i = 2:L
 z(i,1) = cell2mat(a(i-1,1)) * cell2mat(Theta(i-1,1))';
 _a = sigmoid(cell2mat(z(i,1)));
 if (i < L)
  a(i,1) = [ones(m,1) _a];
 else 
  a(i,1) = _a; 
 endif 
endfor 
 
%a1 = [ones(m,1) X];
%z2 = a1 * Theta1';
%a2 = sigmoid(z2);
%a2 = [ones(m,1) a2];
%z3 = a2 * Theta2';
%a3 = sigmoid(z3);

yVect = zeros(m,NNMeta.NNArchVect(L));
for i = 1:NNMeta.NNArchVect(L)
  yVect(:,i) = (y == i);
endfor

%yVect = zeros(m,num_labels);
%for i = 1:num_labels
%  yVect(:,i) = (y == i);
%endfor

%S = ( -1 * (  log(a3) .* yVect) -1 * ( ( log((1 .- a3))) .* (1 .- yVect) ) );
%J = (1/m) * sum(S(:));

hx = cell2mat(a(L,1));
S = ( -1 * (  log(hx) .* yVect) -1 * ( ( log((1 .- hx))) .* (1 .- yVect) ) );
J = (1/m) * sum(S(:));

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

%d_3 = a3 - yVect; 
%Theta2_grad = Theta2_grad + d_3' * a2;
%d_2 = (d_3 * Theta2); 
%d_2 = d_2(:,2:end);  
%d_2 = d_2 .* sigmoidGradient(z2); 
%Theta1_grad = Theta1_grad + d_2' * a1;  




tai = cell(L-1,1);
for i = 1:(L-1)
  s = size(Theta(i,1));
  tai(i,1) = [zeros(s(1),1),cell2mat(tai(i,1))(:,2:end)];
endfor 



%tai1 = Theta1;
%s1=size(Theta1);
%tai1 = [zeros(s1(1),1),tai1(:,2:end)];

%tai2 = Theta2;
%s2=size(Theta2);
%tai2 = [zeros(s2(1),1),tai2(:,2:end)];


for i = 1:(L-1)
  Theta_grad(i,1) = (1/m) * (cell2mat(Theta_grad(i,1)) .+ (lambda)*cell2mat(tai(i,1)));
endfor 


%Theta2_grad = (1/m) * (Theta2_grad .+ (lambda)*tai2);
%Theta1_grad = (1/m) * (Theta1_grad .+ (lambda)*tai1);

%---------------------------- REGULARIZATION 
regTerm = 0; 

for i = 1:(L-1)
  tr = cell2mat(Theta(i,1))(:,2:end);
  tr = tr .* tr;
  regTerm = regTerm + (lambda/(2 * m)) * sum(tr(:));
endfor 


%tr = Theta1(:,2:end);
%tr = tr .* tr;
%regTerm = regTerm + (lambda/(2 * m)) * sum(tr(:));

%tr2 = Theta2(:,2:end);
%tr2 = tr2 .* tr2;
%regTerm = regTerm + (lambda/(2 * m)) * sum(tr2(:));

J = J + regTerm;

% -------------------------------------------------------------

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];

grad = [];
for i = fliplr(1:L-1)
  grad =   [ cell2mat(Theta(i))(:) ;  grad(:) ];
endfor

%v5 = [ cell2mat(c2(3))(:) ;  v5(:) ]


endfunction
