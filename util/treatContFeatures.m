function [X_out] = treatContFeatures(X,p)

[m, n] = size(X);

X_out = polyFeatures(X, p);
[X_out, mu, sigma] = featureNormalize(X_out);  % Normalize
X_out = [ones(m, 1), X_out];                   % Add Ones

endfunction
