function [foX,mu,sigma] = treatContFeatures_Buff(fiX,foX,p,override=0,_mu=0,_sigma=0,b=10000, _sep=',')

m = countLines(fiX,b);
f = fopen(fiX);
str = fgets(f);mt = length(findstr(str,_sep));fclose(f);
n = mt+1;

#init 
mu = zeros(n,1);
sigma = zeros(n,1);

if (! override)

  ## mu
  X = dlmread(fiX,sep=_sep,[0,0,b-1,n-1]);
  mu = sum(X)/m;

  _m = size(X,1);
  c = _m;
  while ((_m == b) && (c < m) )
    X = dlmread(fiX,sep=_sep,[c,0,c+b-1,n-1]);
    mu += sum(X)/m;
        
    _m = size(X,1);
    c += _m;
  endwhile 
  
  
  ## sigma
  X = dlmread(fiX,sep=_sep,[0,0,b-1,n-1]);
  sigma = sum((X - mu) .^2) / m;  
  
  _m = size(X,1);
  c = _m;
  while ((_m == b) && (c < m) )
    X = dlmread(fiX,sep=_sep,[c,0,c+b-1,n-1]);
    sigma = sum((X - mu) .^2) / m;

    _m = size(X,1);
    c += _m;
  endwhile
  
  ## feature normalizing 
  X = dlmread(fiX,sep=_sep,[0,0,b-1,n-1]);
  [X_out, mu, sigma] = featureNormalize(X,override,mu,sigma);
  X_out = [ones(m, 1), X_out]; % Add Ones
  dlmwrite(foX,X_out);

  _m = size(X,1);
  c = _m;
  while ((_m == b) && (c < m) )
    X = dlmread(fiX,sep=_sep,[c,0,c+b-1,n-1]);
    [X_out, mu, sigma] = featureNormalize(X,override,mu,sigma);
    X_out = [ones(m, 1), X_out]; % Add Ones  

    dlmwrite(foX,X_out,"-append");
    _m = size(X,1);
    c += _m;
  endwhile
  
else
  mu = _mu;
  sigma = _sigma;

  ## feature normalizing                                                                                                                                                                                  
  X = dlmread(fiX,sep=_sep,[0,0,b-1,n-1]);
  [X_out, mu, sigma] = featureNormalize(X,override,mu,sigma);
  X_out = [ones(m, 1), X_out]; % Add Ones
  dlmwrite(foX,X_out);

  _m = size(X,1);
  c = _m;
  while ((_m == b) && (c < m) )
    X = dlmread(fiX,sep=_sep,[c,0,c+b-1,n-1]);
    [X_out, mu, sigma] = featureNormalize(X,override,mu,sigma);
    X_out = [ones(m, 1), X_out]; % Add Ones

    dlmwrite(foX,X_out,"-append");
    _m = size(X,1);
    c += _m;
  endwhile

endif

endfunction
