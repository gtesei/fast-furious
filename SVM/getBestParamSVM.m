function [C , gamma , epsilon , nu] = getBestParamSVM (type = 1 , Xtrain, ytrain)
  ## type = 1 - SVR

  C = 0; gamma = 0; epsilon = 0; nu = 0;  
  

  if ( type == 1)
    ## estimating C
    C = (max(ytrain) - min(ytrain));

    #estimating gamma                                                                                
    [x,mu,sigma] = featureNormalize(Xtrain);
    m = size(x,1);
    n = floor(0.5 * m);
    index = ceil(rand(1,n) * m )';
    index2 = ceil(rand(1,n) * m)';
    temp = x(index,:) - x(index2,:);
    dist = sum((temp .* temp)')';
    dist = dist(dist != 0);
    gamma = (quantile(dist ,[0.9 0.5 0.1]) .^ -1);
  endif 

endfunction 
