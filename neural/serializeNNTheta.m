function [_dir] = serializeNNTheta(Theta,useTimestamp = 1, dPrefix="Theta",fPrefix="Theta", fExt="zat", rDir=".")

  ts = strftime ("%d_%m_%Y-%H%M", localtime (time ()));
  
  _dir = "";
  if (useTimestamp)
    _dir = [rDir "/" dPrefix ts];
  else 
    _dir = [rDir "/" dPrefix];
  endif 
  
  if (exist(_dir,'dir'))
    error("output directory already exist!");
  endif
  
  [status, msg, msgid]  = mkdir (_dir);
  
  for i = 1:size(Theta,1)
      fn = [_dir "/" fPrefix num2str(i) "." fExt]; 
      %printf("writing %s",fn);
      dlmwrite( fn , cell2mat(Theta(i,1))); 
  endfor  
  
endfunction