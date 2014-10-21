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

######
srate = 1000;
t = 0: 1/ srate: 10;
n = length( t);
csw = exp( 1i* 2* pi* t);% csw = complex sine wave
plot3( t, real( csw), imag( csw))
xlabel(' time'),
ylabel(' real part')
xlabel(' time'),
ylabel(' real part')
zlabel(' imaginary part')
rotate3d % active click-and-drag in Matlab

########  definition
signal = 2* sin( 2* pi* 3* t + pi/ 2);
fouriertime = (0: n-1)/ n; % n is length( t)
signalX = zeros( size( signal));
for fi = 1: length( signalX)
 csw = exp(-1i* 2* pi*( fi-1)* fouriertime);
signalX( fi) = sum( csw.* signal )/ n;
end



##################
srate = 1000;
t = 0: 1/ srate: 5;
n = length( t);
a =[ 10 2 5 8];
f =[ 3 1 6 12];
swave = zeros( size( t));
for i = 1: length( a)
 swave = swave + a( i)* sin( 2* pi* f( i)* t);
end % Fourier transform

 swaveX = fft( swave)/ n;
 hz = linspace( 0, srate/ 2, floor( n/ 2) + 1); % plot
 subplot( 211), plot( t, swave)
 xlabel(' Time (s)'),
 ylabel(' amplitude')

 subplot( 212)
plot( hz, 2* abs( swaveX( 1: length( hz))))
 set( gca,'xlim',[ 0 max( f)* 1.3]);
 xlabel(' Frequencies (Hz)'),
ylabel(' amplitude')

#############
swaveN = swave + randn( size( swave))* 20;
swaveNX = fft( swaveN)/ n;
subplot( 211),
plot( t, swaveN)
xlabel(' Time (s)'),
ylabel(' amplitude')

subplot( 212)
plot( hz, 2* abs( swaveNX( 1: length( hz))))
set( gca,' xlim',[ 0 max( f)* 1.3]);
xlabel(' Frequencies (Hz)'),
ylabel(' amplitude')


nyquistfreq = srate/ 2;
hz = linspace( 0, nyquistfreq, floor( n/ 2) + 1);


subplot( 211)
plot( hz, 2* abs( signalX( 1: length( hz)))),
hold on
plot( hz, 2* abs( signalXF( 1: length( hz))),' r')
xlabel(' Frequencies (Hz)')
ylabel(' Amplitude')
set( gca,'xlim',[ 0 10]);

subplot( 212)
plot( hz, angle( signalX( 1: floor( n/ 2) + 1)))
xlabel(' Frequencies (Hz)')
ylabel(' Phase (radians)');



#########
srate = 1000;
t = 0: 1/ srate: 5;
n = length(t);
a =[ 10 2 5 8];
f =[ 3 1 6 12];
swave = zeros( size( t));
for i = 1: length( a)
 swave = swave + a( i)* sin( 2* pi* f( i)* t);
end % Fourier transform

swaveX = fft( swave)/ n;
hz = linspace( 0, srate/ 2, floor( n/ 2) + 1);

% plot
subplot( 211),
plot( t, swave)
set( gca,'xlim',[ 0 5]);
xlabel(' Time (s)'),
ylabel(' amplitude')

subplot( 212)
plot( hz, 2* abs( swaveX( 1: length( hz))))
set( gca,'xlim',[ 0 max( f)* 1.3]);
xlabel(' Frequencies (Hz)'),
ylabel(' amplitude')

###
[junk, tenHzidx] = min(abs(hz-10));
tenHzidx = dsearchn( hz', 10);

####
frex_idx = sort( dsearchn( hz', f'));
requested_frequences = 2* abs( swaveX( frex_idx));
bar( requested_frequences)
xlabel(' Frequencies (Hz)'),
ylabel(' Amplitude')

##### extracting power on bands 10 30 60 hz
zeroHzidx = dsearchn( hz', 0);
tenHzidx = dsearchn( hz', 10);
thirtyHzidx = dsearchn( hz', 30);
sixtyHzidx = dsearchn( hz', 60);
tailHzidx = dsearchn( hz', 1000);
frex_idx = [tenHzidx thirtyHzidx sixtyHzidx tailHzidx];
requested_frequences = 2* abs( swaveX( frex_idx));
bar( requested_frequences)
xlabel(' Frequencies (Hz)'),
ylabel(' Amplitude')
                     
power_timedomain = sum(abs(swave).^2) /length(swave)
power_freqdomain  =  sum(swaveX.*conj(swaveX))
                     
p1 = 2*sum(swaveX(zeroHzidx:tenHzidx).*conj(swaveX(zeroHzidx:tenHzidx)))
p2 = 2*sum(swaveX(tenHzidx:thirtyHzidx).*conj(swaveX(tenHzidx:thirtyHzidx)))
p3 = 2*sum(swaveX(thirtyHzidx:sixtyHzidx).*conj(swaveX(thirtyHzidx:sixtyHzidx)))
p4 = 2*sum(swaveX(sixtyHzidx:tailHzidx).*conj(swaveX(sixtyHzidx:tailHzidx)))
                     
p1 + p2 + p3 + p4

bar( [p1 p2 p3 p4] )
xlabel(' Frequencies bands (Hz)'),
ylabel(' Power')
                     
##### find min freq t.c. che cattura 50% power segnale 
nyquistfreq = srate/ 2;
f_50 = -1;
for fr = 0:1:nyquistfreq
  Hzidx = dsearchn( hz', fr);
  p_fr = 2*sum(swaveX(zeroHzidx:Hzidx).*conj(swaveX(zeroHzidx:Hzidx)));
  if (p_fr >= power_timedomain * 0.5 )
    f_50 = fr;
    break;
  endif
endfor

f_50
                   
##### autocorrelazione 
pkg load signal;
xcorr(swave);

                
                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                 



  
  
  