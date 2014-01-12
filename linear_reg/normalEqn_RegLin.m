function [theta] = normalEqn_RegLin(X,y,lambda=0) 
  
  [m,n] = size(X);
  f = zeros(n);
  f(1,1) = 1;

  theta = pinv(X'*X+ lambda * (eye(n)-f) )*X'*y;

endfunction 