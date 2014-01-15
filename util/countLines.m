function [ctr] = countLines(fn,b=10000)                                       

  ctr = 0;

  f = fopen(fn, 'rt');
  ln = fskipl(f,b);
  while (ln > 0) 
    ctr += ln; 
    ln= fskipl(f,b);
  endwhile
  fclose(f);                                                                                           

endfunction  