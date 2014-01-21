function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
m = size(X, 1);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%------------ FORWARD PROP 
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

yVect = zeros(m,num_labels);
for i = 1:num_labels
  yVect(:,i) = (y == i);
endfor

S = ( -1 * (  log(a3) .* yVect) -1 * ( ( log((1 .- a3))) .* (1 .- yVect) ) );
J = (1/m) * sum(S(:));

%-------------BACK PROP

d_3 = a3 - yVect; 
Theta2_grad = Theta2_grad + d_3' * a2;
d_2 = (d_3 * Theta2); 
d_2 = d_2(:,2:end);  
d_2 = d_2 .* sigmoidGradient(z2); 
Theta1_grad = Theta1_grad + d_2' * a1;  


tai1 = Theta1;
s1=size(Theta1);
tai1 = [zeros(s1(1),1),tai1(:,2:end)];

tai2 = Theta2;
s2=size(Theta2);
tai2 = [zeros(s2(1),1),tai2(:,2:end)];


Theta2_grad = (1/m) * (Theta2_grad .+ (lambda)*tai2);
Theta1_grad = (1/m) * (Theta1_grad .+ (lambda)*tai1);

%---------------------------- REGULARIZATION 
regTerm = 0; 

tr = Theta1(:,2:end);
tr = tr .* tr;
regTerm = regTerm + (lambda/(2 * m)) * sum(tr(:));

tr2 = Theta2(:,2:end);
tr2 = tr2 .* tr2;
regTerm = regTerm + (lambda/(2 * m)) * sum(tr2(:));

J = J + regTerm;

% -------------------------------------------------------------

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

endfunction
