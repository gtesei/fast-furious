function [X_out,mu,sigma] = treatContFeatures_Buff(fX,ciX,ceX,p,override=0,_mu=0,_sigma=0,b=10000, _sep=',')

m = countLines(fX,b);

X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
y = dlmread(fy,sep=_sep,[0,ciy,b-1,cey]);
X_out = polyFeatures(X, p);

if (! override)
  mu = TODO - calcolare mu su tutto X in maniera buffered 
  sigma = TODO - calacolare su tutto X in maniera buffered 
  [X_out, mu, sigma] = featureNormalize(X_out,override,mu,sigma);  
else
  mu = _mu;
  sigma = _sigma;
  [X_out, mu, sigma] = featureNormalize(X_out,override,mu,sigma);  
endif

X_out = [ones(m, 1), X_out]; % Add Ones

endfunction
