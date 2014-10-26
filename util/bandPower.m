function [p0,p1,p2,p3,p4,p5,p6,p7,P] = bandPower (fs,T,x,debug=0) 
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
  hzIdx6 = dsearchn( hz', 90);
  hzIdx7 = dsearchn( hz', 150);
  hzIdxTail = dsearchn( hz', nyquistfreq);
        
  P =  sum(swaveX.*conj(swaveX));
        
  p0 = 2*sum(swaveX(1:hzIdx0).*conj(swaveX(1:hzIdx0))) / P;
  p1 = 2*sum(swaveX(hzIdx0:hzIdx1).*conj(swaveX(hzIdx0:hzIdx1))) / P;
  p2 = 2*sum(swaveX(hzIdx1:hzIdx2).*conj(swaveX(hzIdx1:hzIdx2))) / P;
  p3 = 2*sum(swaveX(hzIdx2:hzIdx3).*conj(swaveX(hzIdx2:hzIdx3))) / P;
  p4 = 2*sum(swaveX(hzIdx3:hzIdx4).*conj(swaveX(hzIdx3:hzIdx4))) / P;
  p5 = 2*sum(swaveX(hzIdx4:hzIdx5).*conj(swaveX(hzIdx4:hzIdx5))) / P;
  p6 = 2*sum(swaveX(hzIdx5:hzIdx6).*conj(swaveX(hzIdx5:hzIdx6))) / P;
  p7 = 2*sum(swaveX(hzIdx6:hzIdx7).*conj(swaveX(hzIdx6:hzIdx7))) / P;
  pTail = 2*sum(swaveX(hzIdx7:hzIdxTail).*conj(swaveX(hzIdx7:hzIdxTail))) /P;
  
  if (debug)
    power_timedomain = sum(abs(x).^2) /length(x);
    if ( (p0+ p1 + p2 + p3 + p4 + p5 + p6 + p7 + pTail - 1) > 0.01)
      printf("warning: the sum of  signal bands' power is not equal to the signal power!");
    endif 
    if ((power_timedomain-P)/P > 0.01) 
      printf("warning: signal power time domain different from signal power frequency domanin!");
    endif 
  endif 

endfunction
