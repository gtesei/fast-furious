function [f_50] = findSpectralEdgeFrequency (fs,T,x)
  % fs = Sample frequency (Hz)
  % T = secs sample
  % x = signa
  
  t = 0:1/fs:T-1/fs;      
  n = length(t);
  
  swaveX = fft(x)/ n;
  hz = linspace( 0, fs/2, floor(n/2) + 1);
  
  power_freqdomain =  sum(swaveX.*conj(swaveX));
  
  nyquistfreq = fs/2;

  f_50 = -1;
  for fr = 0:1:nyquistfreq
    Hzidx = dsearchn( hz', fr);
    p_fr = 2*sum(swaveX(1:Hzidx).*conj(swaveX(1:Hzidx)));
    if (p_fr >= power_freqdomain * 0.5 )
      f_50 = fr;
      break;
    endif
  endfor
  
endfunction 