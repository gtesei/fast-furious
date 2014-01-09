function [X_out,mu,sigma] = treatContFeatures(X,p,override=0,_mu=0,_sigma=0)

[m, n] = size(X);

X_out = polyFeatures(X, p);

if (! override)
  [X_out, mu, sigma] = featureNormalize(X_out);  % Normalize
else
  mu = _mu;
  sigma = _sigma;
  [X_out, mu, sigma] = featureNormalize(X_out,override,mu,sigma);  % Normalize
endif

X_out = [ones(m, 1), X_out]; % Add Ones

endfunction
