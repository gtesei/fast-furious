function [pred] = NNPredictRegEGS(NNMeta, Theta , X , featureScaled = 0)


m = size(X, 1);

L = length(NNMeta.NNArchVect);

%-------- EGS
n = size( X ,2);
if (!featureScaled)
  n = n + 1;
end
links = ceil( n  ^ (1/(L-1)) );
%--------


h = cell(L-1,1);
for i = 1:L-1
  _nn = size( cell2mat(Theta(i,1)) ,2);
  _mm = size( cell2mat(Theta(i,1)) ,1);
  _pe = cell2mat(Theta(i,1)) .* EGS( _mm , _nn , links , i  );


  if (featureScaled & i == 1)
    h(i,1) = X * _pe';
  elseif (i == 1)
    h(i,1) = [ones(m, 1) X] * _pe';
  else 
    h(i,1) = [ones(m, 1) cell2mat(h(i-1,1))] * _pe';
  endif

endfor




pred = cell2mat(h(L-1,1));
endfunction 
