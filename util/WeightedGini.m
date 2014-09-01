function [gini] = WeightedGini (solution, weights, submission)

  df = [solution weights submission];
  df = sortrows(df,-3);
  random = cumsum(df(:,2)/sum(df(:,2)));
  totalPositive = sum( df(:,1) .* df(:,2) );
  cumPosFound = cumsum( df(:,1) .* df(:,2) );
  Lorentz = cumPosFound / totalPositive; 
  n = size(df,1);
  gini =  sum( Lorentz(2:end) .* random(1:(n-1)) ) - sum( Lorentz(1:(n-1)) .* random(2:end) ); 

end
