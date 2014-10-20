pkg load signal;

  %%% example of signal 
  T = 10;                                  % 10 sec sample
  fs = 100;                                % Sample frequency (Hz)
  t = 0:1/fs:T-1/fs;                       % 10 sec sample
  x = (1.3)*sin(2*pi*15*t) ...             % 15 Hz component
   + (1.7)*sin(2*pi*40*(t-2)) ...         % 40 Hz component
   + 2*randn(size(t));                    %  noise;

    plot(t,x)
    xlabel('time (1/100 sec)')
    ylabel('signal')
    title('Time Domain');
  
  %%% FFT 
  m = length(x);          % Window length
  n = pow2(nextpow2(m));  % Transform length
  y = fft(x,n);           % DFT
  f = (0:n-1)*(fs/n);     % Frequency range
  power = y.*conj(y)/n;   % Power of the DFT
  fm = min( find( power == max(power) ) * (fs/n) );
  plot(f,power)
  xlabel('Frequency (Hz)')
  ylabel('Power')
  title('Periodogram');
  
  y = fft(x);
  power_timedomain = sum(abs(x).^2) /length(x)
  power_freqdomain  =  sum(y.*conj(y))/(length(y)^2) 
  
  f = (0:length(y)-1)*(fs/length(y));
  power = y.*conj(y)/(length(y)^2) ;
  plot(f,power)
  xlabel('Frequency (Hz)')
  ylabel('Power')
  title('Periodogram');
    
  
  
  