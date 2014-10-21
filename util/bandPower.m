function [p1,p2,p3,p4,p5,P] = bandPower (fs,T,x,debug=0) 
  % fs = Sample frequency (Hz)
  % T = secs sample
  % x = signa
  
  t = 0:1/fs:T-1/fs;      
  n = length(t);
  
  swaveX = fft(x)/ n;
  hz = linspace( 0, fs/2, floor(n/2) + 1);
  
  nyquistfreq = fs/2;
  
  hzIdx0 = dsearchn( hz', 0.5);
  hzIdx1 = dsearchn( hz', 4);
  hzIdx2 = dsearchn( hz', 8);
  hzIdx3 = dsearchn( hz', 13);
  hzIdx4 = dsearchn( hz', 30);
  hzIdx5 = dsearchn( hz', 48);
  hzIdxTail = dsearchn( hz', nyquistfreq);
  
  p0 = 2*sum(swaveX(1:hzIdx0).*conj(swaveX(1:hzIdx0)));
  p1 = 2*sum(swaveX(hzIdx0:hzIdx1).*conj(swaveX(hzIdx0:hzIdx1)));
  p2 = 2*sum(swaveX(hzIdx1:hzIdx2).*conj(swaveX(hzIdx1:hzIdx2)));
  p3 = 2*sum(swaveX(hzIdx2:hzIdx3).*conj(swaveX(hzIdx2:hzIdx3)));
  p4 = 2*sum(swaveX(hzIdx3:hzIdx4).*conj(swaveX(hzIdx3:hzIdx4)));
  p5 = 2*sum(swaveX(hzIdx4:hzIdx5).*conj(swaveX(hzIdx4:hzIdx5)));
  pTail = 2*sum(swaveX(hzIdx5:hzIdxTail).*conj(swaveX(hzIdx5:hzIdxTail)));
  
  P =  sum(swaveX.*conj(swaveX));
  
  if (debug)
    power_timedomain = sum(abs(x).^2) /length(x);
    if ( ((p0 + p1 + p2 + p3 + p4 + p5 + pTail) - P)/P > 0.01)
      printf("warning: the sum of  signal bands' power is not equal to the signal power!");
    endif 
    if ((power_timedomain-P)/P > 0.01) 
      printf("warning: signal power time domain different from signal power frequency domanin!");
    endif 
  endif 

endfunction
