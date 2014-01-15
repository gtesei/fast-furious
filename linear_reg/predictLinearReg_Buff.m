function [y] = predictLinearReg_Buff(fX,ciX,ceX,theta,b=10000,_sep=',')

 m = countLines(fX,b);
 y = zeros(m,1); 

 X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
 y(1:min(m,b),:) = X * theta;
 c = min(b,m);
 while ((size(X,1) == b) && (c < m) )
   X = dlmread(fX,sep=_sep,[c,ciX,c+b-1,ceX]);
   _b = size(X,1);
   y(c+1:c+_b,:) = X * theta;
   c += _b;
 endwhile

endfunction 
