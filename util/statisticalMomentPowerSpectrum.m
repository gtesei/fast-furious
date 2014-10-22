function [m1,m2,m3,m4] = statisticalMomentPowerSpectrum (fs,T,x) 
  % fs = Sample frequency (Hz)
  % T = secs sample
  % x = signa
  
  t = 0:1/fs:T-1/fs;      
  n = length(t);
  
  swaveX = fft(x)/ n;
  hz = linspace( 0, fs/2, floor(n/2) + 1);
  
  nyquistfreq = fs/2;

  hzIdxTail = dsearchn( hz', nyquistfreq);
  P =  sum(swaveX.*conj(swaveX));
  
  m1 = (1/P)*2*sum(swaveX(1:hzIdxTail).*conj(swaveX(1:hzIdxTail)).* (1:hzIdxTail));
  m2 = (1/P)*2*sum(swaveX(1:hzIdxTail).*conj(swaveX(1:hzIdxTail)).* ((1:hzIdxTail).*(1:hzIdxTail)) );
  m3 = (1/P)*2*sum(swaveX(1:hzIdxTail).*conj(swaveX(1:hzIdxTail)).* (1:hzIdxTail).*(1:hzIdxTail).*(1:hzIdxTail)  );
  m4 = (1/P)*2*sum(swaveX(1:hzIdxTail).*conj(swaveX(1:hzIdxTail)).* (1:hzIdxTail).*(1:hzIdxTail).*(1:hzIdxTail).*(1:hzIdxTail)  );
  
endfunction
