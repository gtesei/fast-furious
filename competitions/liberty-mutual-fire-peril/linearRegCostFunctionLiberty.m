function [J, grad] = linearRegCostFunctionLiberty(X, y, theta, lambda , var11 )

m = length(y);
J = 0;
grad = zeros(size(theta));

% ====================== 
tai = [0;theta(2:end,1)];
                 
hx = X * theta;
gini = NormalizedWeightedGini(y,var11,hx);
norm = 0;
if (lambda > 0)
  norm = ( lambda / (2*m) ) * (tai' * tai);
endif 
J = (1 - gini) + norm;
printf("gini = %f , J = %f , norm = %f \n",gini,J,norm);
                                                                      
% ======= calcola grad numericamente
n = size(theta,1);
e = 1e-4;
E = diag(e * ones(n,1));
                         
Theta1 = zeros(n,n);
Theta2 = zeros(n,n);                         
for i = 1:n 
  Theta1(:,i) = theta;
  Theta2(:,i) = theta; 
endfor                          

Theta1 = Theta1 + E;
Theta2 = Theta2 - E;

H1 = X * Theta1; 
H2 = X * Theta2; 

####### WG1 = NormalizedWeightedGiniVect (y, var11, H1);

  %%%% submissions
  nn = size(H1,2);
  giniVect = zeros(nn,1);

  for i = 1:nn
   df = [y var11 H1(:,i)];

   df = sortrows(df,-3);
   random = cumsum(df(:,2)/sum(df(:,2)));
   totalPositive = sum( df(:,1) .* df(:,2) );
   cumPosFound = cumsum( df(:,1) .* df(:,2) );
   Lorentz = cumPosFound / totalPositive;
   k = size(df,1);
   gini =  sum( Lorentz(2:end) .* random(1:(k-1)) ) - sum( Lorentz(1:(k-1)) .* random(2:end) );
   giniVect(i) = gini;
  endfor


  %%%% solution
  df = [y var11 y];

  df = sortrows(df,-3);
  random = cumsum(df(:,2)/sum(df(:,2)));
  totalPositive = sum( df(:,1) .* df(:,2) );
  cumPosFound = cumsum( df(:,1) .* df(:,2) );
  Lorentz = cumPosFound / totalPositive;
  k = size(df,1);
  giniSolution =  sum( Lorentz(2:end) .* random(1:(k-1)) ) - sum( Lorentz(1:(k-1)) .* random(2:end) );

  %%%% return
WG1 = giniVect / giniSolution;



####### WG2 = NormalizedWeightedGiniVect (y, var11, H2);

  %%%% submissions
  nn = size(H2,2);
  giniVect = zeros(nn,1);

  for i = 1:nn
   df = [y var11 H2(:,i)];

   df = sortrows(df,-3);
   random = cumsum(df(:,2)/sum(df(:,2)));
   totalPositive = sum( df(:,1) .* df(:,2) );
   cumPosFound = cumsum( df(:,1) .* df(:,2) );
   Lorentz = cumPosFound / totalPositive;
   k = size(df,1);
   gini =  sum( Lorentz(2:end) .* random(1:(k-1)) ) - sum( Lorentz(1:(k-1)) .* random(2:end) );
   giniVect(i) = gini;
  endfor


  %%%% return
WG2= giniVect / giniSolution;

LOSS1 = ones(n,1) - WG1;
LOSS2 = ones(n,1) - WG2;

if (lambda > 0)
  Theta1(1,:) = zeros(1,n);
  Theta2(1,:) = zeros(1,n);

  LOSS1 = LOSS1 + (lambda/(2*m)) * diag(Theta1 * Theta1'); 
  LOSS2 = LOSS2 + (lambda/(2*m)) * diag(Theta2 * Theta2');  
endif  

numgrad = (1 / (2*e)) * (LOSS1 - LOSS2); 
grad = numgrad(:);

fflush(stdout);
endfunction 
