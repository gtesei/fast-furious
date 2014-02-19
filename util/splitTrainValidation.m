function [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X,y,perc_train) 

 [m,n] = size(X);
 [_m,_n] = size(y);
 
 if (m != _m)
   printf("m(features)=%i   _m(classes)=%i \n",m,_m); 
   error ("the length of y must be equal to the length of X!");
 elseif (perc_train > 1 || perc_train < 0)
   error ("perc_train must belong to the range [0,1]!");
 endif
 
 mtrain = floor(m * perc_train);
 mval = m - mtrain;
 
 Xtrain = X(1:mtrain,:); 
 Xval = X(mtrain+1:end,:);
 ytrain = y(1:mtrain,:);
 yval = y(mtrain+1:end,:);
 
endfunction 
