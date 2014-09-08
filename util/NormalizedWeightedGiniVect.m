function [nginiVect] = NormalizedWeightedGiniVect (solution, weights, submissions)

  %%%% submissions
  n = size(submissions,2);
  giniVect = zeros(n,1);

  for i = 1:n
   df = [solution weights submissions(:,i)];

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
  df = [solution weights solution];
  df = sortrows(df,-3);
  random = cumsum(df(:,2)/sum(df(:,2)));
  totalPositive = sum( df(:,1) .* df(:,2) );
  cumPosFound = cumsum( df(:,1) .* df(:,2) );
  Lorentz = cumPosFound / totalPositive;
  k = size(df,1);
  giniSolution =  sum( Lorentz(2:end) .* random(1:(k-1)) ) - sum( Lorentz(1:(k-1)) .* random(2:end) );

  %%%% return
  nginiVect = giniVect / giniSolution;

end
