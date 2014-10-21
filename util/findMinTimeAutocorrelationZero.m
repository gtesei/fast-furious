function [min_tau] = findMinTimeAutocorrelationZero (x,fs)
 pkg load signal;

 [cz,lag] = xcorr(x,'coeff');
 [mC,Im] = min(abs(cz) == 0); 
 min_tau = lag(Im)/fs;

endfunction 