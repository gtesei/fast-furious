function [X_poly] = polyFeatures(X, p)
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

[m, n] = size(X);
X_poly = zeros(m, (n*p));

X_poly(:,1:n) = X;
for i = 2:p
        X_poly(:,(i-1)*n+1:i*n) = X_poly(:,(i-2)*n+1:(i-1)*n) .* X_poly(:,1:n); 
end

endfunction
