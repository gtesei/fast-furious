function [X_norm, mu, sigma] = featureNormalize(X,override=0,_mu=0,_sigma=0)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

if (! override)
  mu = mean(X);
  %sigma = max(X) - min(X);
  sigma = std(X); % better performances 
  if ( ! all(sigma) )
   sigma(find(sigma==0)) = 1; %preventing NaN
  endif
else
  mu = _mu;
  sigma = _sigma;
endif

X_norm = bsxfun(@minus, X, mu); 
X_norm = bsxfun(@rdivide, X_norm, sigma);

endfunction 
