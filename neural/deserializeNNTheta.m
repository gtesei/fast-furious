function [Theta] = deserializeNNTheta(NNMeta,_dir,fPrefix="Theta", fExt="zat")
  
  if (! exist(_dir,'dir'))
    error("output directory already exist!");
  endif
  
  L = length(NNMeta.NNArchVect); 
  Theta = cell(L-1,1);
  for i = 1:(L-1)
       fn = [_dir "/" fPrefix num2str(i) "." fExt];
       Theta(i,1) = dlmread(fn);
  endfor  
    
endfunction