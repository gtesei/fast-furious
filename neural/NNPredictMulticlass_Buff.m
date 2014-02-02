function [y] = NNPredictMulticlass_Buff(NNMeta,fX,ciX,ceX,Theta,b=10000,_sep=',',featureScaled = 0)

 m = countLines(fX,b);
 y = zeros(m,1); 
 L = length(NNMeta.NNArchVect);
 h = cell(L-1,1);
 
 #################
 X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
 for i = 1:L-1
  if (featureScaled & i == 1)
    h(i,1) = sigmoid(X * cell2mat(Theta(i,1))');
  elseif (i == 1)
    h(i,1) = sigmoid([ones(m, 1) X] * cell2mat(Theta(i,1))');
  else 
    h(i,1) = sigmoid([ones(m, 1) cell2mat(h(i-1,1))] * cell2mat(Theta(i,1))');
  endif
 endfor 
 [dummy, y(1:min(m,b),:)] = max(cell2mat(h(L-1,1)), [], 2);
 #################

 c = min(b,m);
 while ((size(X,1) == b) && (c < m) )
   X = dlmread(fX,sep=_sep,[c,ciX,c+b-1,ceX]);
   _b = size(X,1);

   ##y(c+1:c+_b,:) = X * theta;
   #################
   X = dlmread(fX,sep=_sep,[0,ciX,b-1,ceX]);
   for i = 1:L-1
    if (featureScaled & i == 1)
      h(i,1) = sigmoid(X * cell2mat(Theta(i,1))');
    elseif (i == 1)
      h(i,1) = sigmoid([ones(m, 1) X] * cell2mat(Theta(i,1))');
    else 
      h(i,1) = sigmoid([ones(m, 1) cell2mat(h(i-1,1))] * cell2mat(Theta(i,1))');
    endif
   endfor 
   [dummy, y(c+1:c+_b,:)] = max(cell2mat(h(L-1,1)), [], 2);
   #################
  
  c += _b;
 endwhile

endfunction 
