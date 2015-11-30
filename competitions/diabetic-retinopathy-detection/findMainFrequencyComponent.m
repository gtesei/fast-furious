function [fm] = findMainFrequencyComponent (fs,T,x,doPlot)
  % fs = Sample frequency (Hz)
  % T = secs sample
  % x = signal
  
  t = 0:1/fs:T-1/fs;                                  
  m = length(x);          % Window length
  n = pow2(nextpow2(m));  % Transform length
  y = fft(x,n);           % DFT
  f = (0:n-1)*(fs/n);     % Frequency range
  power = y.*conj(y)/n;   % Power of the DFT
  fm = min( find( power == max(power) ) * (fs/n) );
  
  if (doPlot)
    plot(f,power)
    xlabel('Frequency (Hz)')
    ylabel('Power')
    title('Periodogram');
  end
  
end

