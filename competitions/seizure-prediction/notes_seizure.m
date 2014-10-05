
%%%%%%%%%%%%%%%%%  
clear;
fs = 95;                                % Sample frequency (Hz)
t = 0:1/fs:12-1/fs;                      % 12 sec sample
x = (1.3)*sin(2*pi*15*t);                % 15 Hz component
m = length(x);          % Window length
n = pow2(nextpow2(m));  % Transform length
y = fft(x,n);           % DFT
f = (0:n-1)*(fs/n);     % Frequency range
power = y.*conj(y)/n;   % Power of the DFT
plot(f,power)
xlabel('Frequency (Hz)')
ylabel('Power')
title('Periodogram');
fm = min( find( power == max(power) ) * (fs/n) );
printf("fm=%f \n",fm);


%%%%%%%%%%%%%%%%%  testing signal x as a vector
clear;
fs = 95;                                % Sample frequency (Hz)
t = 0:1/fs:12-1/fs;                      % 12 sec sample
for ff = 1:25
  x = (1.3)*sin(2*pi*ff*t);                % 15 Hz component 
  m = length(x);          % Window length
  n = pow2(nextpow2(m));  % Transform length
  y = fft(x,n);           % DFT
  f = (0:n-1)*(fs/n);     % Frequency range
  power = y.*conj(y)/n;   % Power of the DFT
  plot(f,power)
  xlabel('Frequency (Hz)')
  ylabel('Power')
  title('Periodogram');
  fm = min( find( power == max(power) ) * (fs/n) );
  printf("ff=%f - fm=%f \n",ff,fm);
endfor 

%%%%%%%%%%%%%%%%%  testing findMainFrequencyComponent
menv;
fs = 95;                                % Sample frequency (Hz)
T = 12;
t = 0:1/fs:T-1/fs; 

for ff = 1:25
  x = (1.3)*sin(2*pi*ff*t); 
  [fm] = findMainFrequencyComponent (fs,T,x',doPlot=0);
  printf("ff=%f - fm=%f \n",ff,fm);
endfor

########## 
x = [1 2 3]';  
q = quantile (x, [0, 1]);  
q = quantile (x, [0.25 0.5 0.75]);


########## 
dirname = [curr_dir "/dataset/seizure-prediction/" ds "_digest/"];
mkdir(dirname);

########## 
fn = [curr_dir "/dataset/seizure-prediction/aa.zat"];
matrix = [1 2 3 4; 5 6 7 8; 9 10 11 12];
dlmwrite(fn,matrix);
matrix
matrix2 = dlmread(fn);
matrix2
matrix - matrix2

##########
s = 1; 
d = 5; 
if (s > 0 || d > 4)
  printf("pippo \n");
endif 

##########
matrix = [1 2 3 4 5 6 7 8; 9 10 11 12 13 14 16 16];
vect = [-1 -2 -3];
matrix(1,2:4) = vect;





  



