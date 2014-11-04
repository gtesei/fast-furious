
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spectral coherence 
srate = 1000; 
t = 0: 1/ srate: 9-1/ srate; 
n = length( t); 

% create signals 
f = [10 14 8]; 
k1 =( f( 1)/ srate)* 2* pi/ f( 1); 

sigA = sin( 2* pi.* f( 1).* t + k1* cumsum( 5* randn( 1, n))) + randn( size( t)); 
sigB = sin( 2* pi.* f( 2).* t + k1* cumsum( 5* randn( 1, n))) + sigA; 

sigA = sigA + sin( 2* pi.* f( 3).* t + k1* cumsum( 5* randn( 1, n))); 

% ---------------- signA e signB sono i segnali da cui calcolare la spectral cohertence 

% show power of each channel 
hz = linspace( 0, srate/ 2, floor( n/ 2) + 1); 
sigAx = fft( sigA)/ n; 
sigBx = fft( sigB)/ n; 
subplot( 211)
plot( hz, 2* abs( sigAx( 1: length( hz)))), hold on 
plot( hz, 2* abs( sigBx( 1: length( hz))),' r') 

% spectral coherence 
specX = abs( sigAx.* conj( sigBx)).^2; 
spectcoher = specX./( sigAx.* sigBx); 

powerCoher = spectcoher.*conj(spectcoher)/n;   % Power of the DFT
fm = min( find( powerCoher == max(powerCoher) ) * (srate/n) )
pause;  

subplot( 212) 
plot( hz, abs( spectcoher( 1: length( hz))))
pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% applying to the problem 
menv;

########################### CONSTANTS 
INTERICTAL_MODE = 1;
PREICTAL_MODE = 2;
TEST_MODE = 3;

NUMBER_OF_FILES = 2;
########################### loadind data sets and meta-data
printf("|--> generating features set ...\n");

%%dss = ["Dog_1"; "Dog_2"; "Dog_3"; "Dog_4"; "Dog_5"; "Patient_1"; "Patient_2"];
dss = ["Patient_2"];
cdss = cellstr (dss);

printf("|--> found %i data sets ... \n",size(cdss,1));
for i = 1:size(cdss,1)
  printf("|- %i - %s \n",i,cell2mat(cdss(i,1)));
endfor 

for i = 1:size(cdss,1)
  ds = cell2mat(cdss(i,1)); 
  printf("|---> processing %s  ...\n",ds);
  
  %% data files 
  %%pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_interictal_segment*"]);
  pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_interictal_segment*"]);
  interictal_files = glob (pattern_interictal);
  
  %%pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_preictal_segment*"]);
  pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_preictal_segment*"]);
  preictal_files = glob (pattern_preictal);
  
  %%pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_test_segment*"]);
  pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_test_segment*"]);
  test_files = glob (pattern_test);
  
  %%%%%%%%%% main loop  
  for mode = 1:2
    printf("|---> mode = %i (1 = inter,2=preict,3=test) ...\n",mode);
    
    files = test_files;
    if (mode == INTERICTAL_MODE)
      files = interictal_files;
    elseif (mode == PREICTAL_MODE) 
      files = preictal_files;
    endif 
    
    for fi = 1:size(files,1)
      
      if (fi > NUMBER_OF_FILES) 
        break;  
      endif 
      
      fn = cell2mat(files(fi,1));
      printf("|-- processing %s  ...\n",fn);
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      seg = load  (fn);
      names = fieldnames(seg);
      name_seg = cell2mat(names);
      seg_struct = getfield(seg,name_seg);
      
      %% garbage collection .. 
      clear seg; 
      
      seg_struct_names = fieldnames(seg_struct);
      
      name = cell2mat(seg_struct_names(1,1));
      data = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(2,1));
      data_length_sec = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(3,1));
      sampling_fraquency = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(4,1));
      channels = getfield(seg_struct,name);
      
      %% garbage collection .. 
      clear seg_struct;
      
      %%% signal processing ... 
      spectcoher = []; 
      for i = 2:size(data,1)
        
          t = 0:1/sampling_fraquency:data_length_sec-1/sampling_fraquency;      
          n = length(t);
          
          sigAx = spectcoher; 
          if (i == 2)
             sigAx = fft( data(1,:)')/ n;
          endif
         
          sigBx = fft( data(i,:)')/ n; 
          
          % spectral coherence 
          specX = abs( sigAx.* conj( sigBx)).^2; 
          spectcoher = specX./( sigAx.* sigBx); 
          
          powerCoher = spectcoher.*conj(spectcoher)/n;   % Power of the DFT
          fm = min( find( powerCoher == max(powerCoher) ) * (sampling_fraquency/n) );
          
          printf("|---> spectral coherence among signal %i and before has fm = %f \n",i,fm);
          
          
      endfor
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      fflush(stdout);
    endfor 
  endfor 
endfor 

 

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PCA
 % covariance of data 
 v = rand( 10); 
 c = chol( v* v'); 
 n = 10000; 
 % n time points 
 d = randn( n, size( v, 1))* c; 
 % subtract mean and compute covariance ---- d e' il segnale (10000 obs x 10 canali )
 d = bsxfun(@ minus, d, mean( d, 1)); 
 covar = (d'* d)./( n-1); 
 
 imagesc( covar)
 pause;
 
 % compute PCA and eigenvalues (ev) 
 [pc, ev] = eig( covar); 
 % re-sort components 
 #pc = pc(:, end:-1: 1); 
 % extract eigenvalues and covert to % 
 %%ev = diag( ev); ev = 100* ev( end:-1: 1)./ sum( ev); 
 ev = diag( ev); 
 ev = 100* ev./ sum( ev); 
 
 plot( ev,'-o')

 
 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% applying to the problem 
menv;

########################### CONSTANTS 
INTERICTAL_MODE = 1;
PREICTAL_MODE = 2;
TEST_MODE = 3;

NUMBER_OF_FILES = 2;
########################### loadind data sets and meta-data
printf("|--> generating features set ...\n");

%%dss = ["Dog_1"; "Dog_2"; "Dog_3"; "Dog_4"; "Dog_5"; "Patient_1"; "Patient_2"];
dss = ["Patient_2"];
cdss = cellstr (dss);

printf("|--> found %i data sets ... \n",size(cdss,1));
for i = 1:size(cdss,1)
  printf("|- %i - %s \n",i,cell2mat(cdss(i,1)));
endfor 
 
 for i = 1:size(cdss,1)
  ds = cell2mat(cdss(i,1)); 
  printf("|---> processing %s  ...\n",ds);
  
  %% data files 
  %%pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_interictal_segment*"]);
  pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_interictal_segment*"]);
  interictal_files = glob (pattern_interictal);
  
  %%pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_preictal_segment*"]);
  pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_preictal_segment*"]);
  preictal_files = glob (pattern_preictal);
  
  %%pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_test_segment*"]);
  pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_test_segment*"]);
  test_files = glob (pattern_test);
  
  %%%%%%%%%% main loop  
  for mode = 1:2
    printf("|---> mode = %i (1 = inter,2=preict,3=test) ...\n",mode);
    
    files = test_files;
    if (mode == INTERICTAL_MODE)
      files = interictal_files;
    elseif (mode == PREICTAL_MODE) 
      files = preictal_files;
    endif 
    
    for fi = 1:size(files,1)
      
      if (fi > NUMBER_OF_FILES) 
        break;  
      endif 
      
      fn = cell2mat(files(fi,1));
      printf("|-- processing %s  ...\n",fn);
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      seg = load  (fn);
      names = fieldnames(seg);
      name_seg = cell2mat(names);
      seg_struct = getfield(seg,name_seg);
      
      %% garbage collection .. 
      clear seg; 
      
      seg_struct_names = fieldnames(seg_struct);
      
      name = cell2mat(seg_struct_names(1,1));
      data = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(2,1));
      data_length_sec = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(3,1));
      sampling_fraquency = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(4,1));
      channels = getfield(seg_struct,name);
      
      %% garbage collection .. 
      clear seg_struct;
      
      %%% signal processing ... 
      
        
          t = 0:1/sampling_fraquency:data_length_sec-1/sampling_fraquency;      
          n = length(t);
          
          d = bsxfun(@ minus, data', mean( data', 1)); 
          covar = (d'* d)./( n-1);
          
          [pc, ev] = eig( covar); 
          ev = diag( ev); 
          ev = 100* ev./ sum( ev); 
          ev_main = sum(ev > 10);
          printf("|-- numero di PC che spiegano almeno il 10 perc. della variabilita' = %f  ...\n",ev_main);
 
          plot( ev,'-o')
          pause; 
          
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      fflush(stdout);
    endfor 
  endfor 
endfor 



########################### automatic  
menv;

########################### CONSTANTS 
INTERICTAL_MODE = 1;
PREICTAL_MODE = 2;
TEST_MODE = 3;

NUMBER_OF_FILES = 2;
########################### loadind data sets and meta-data
printf("|--> generating features set ...\n");

%%dss = ["Dog_1"; "Dog_2"; "Dog_3"; "Dog_4"; "Dog_5"; "Patient_1"; "Patient_2"];
dss = ["Patient_1"];
cdss = cellstr (dss);

interict_mat = [];
preict_mat = [];

printf("|--> found %i data sets ... \n",size(cdss,1));
for i = 1:size(cdss,1)
  printf("|- %i - %s \n",i,cell2mat(cdss(i,1)));
endfor 
 
 for i = 1:size(cdss,1)
  ds = cell2mat(cdss(i,1)); 
  printf("|---> processing %s  ...\n",ds);
  
  %% data files 
  %%pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_interictal_segment*"]);
  pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_interictal_segment*"]);
  interictal_files = glob (pattern_interictal);
  
  %%pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_preictal_segment*"]);
  pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_preictal_segment*"]);
  preictal_files = glob (pattern_preictal);
  
  %%pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_test_segment*"]);
  pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "_test_segment*"]);
  test_files = glob (pattern_test);
  
  %%%%%%%%%% main loop  
  interict_mat = zeros(size(interictal_files,1),11);
  preict_mat = zeros(size(preictal_files,1),11);
  
  for mode = 1:2
    printf("|---> mode = %i (1 = inter,2=preict,3=test) ...\n",mode);
    
    files = test_files;
    if (mode == INTERICTAL_MODE)
      files = interictal_files;
    elseif (mode == PREICTAL_MODE) 
      files = preictal_files;
    endif 
    
    for fi = 1:size(files,1)
      
      fn = cell2mat(files(fi,1));
      printf("|-- processing %s  ...\n",fn);
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      seg = load  (fn);
      names = fieldnames(seg);
      name_seg = cell2mat(names);
      seg_struct = getfield(seg,name_seg);
      
      %% garbage collection .. 
      clear seg; 
      
      seg_struct_names = fieldnames(seg_struct);
      
      name = cell2mat(seg_struct_names(1,1));
      data = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(2,1));
      data_length_sec = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(3,1));
      sampling_fraquency = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(4,1));
      channels = getfield(seg_struct,name);
      
      %% garbage collection .. 
      clear seg_struct;
      
      %%% signal processing ... 
      
        
          t = 0:1/sampling_fraquency:data_length_sec-1/sampling_fraquency;      
          n = length(t);
          
          d = bsxfun(@ minus, data', mean( data', 1)); 
          covar = (d'* d)./( n-1);
          
          [pc, ev] = eig( covar); 
          ev = diag( ev); 
          ev = 100* ev./ sum( ev); 
          ev_main = sum(ev > 10);
          printf("|-- numero di PC che spiegano almeno il 10 perc. della variabilita' = %f  ...\n",ev_main);
          
          BASE = 5;
          ev_5 = sum(ev > BASE+1);
          ev_6 = sum(ev > BASE+2);
          ev_7 = sum(ev > BASE+3);
          ev_8 = sum(ev > BASE+4);
          ev_9 = sum(ev > BASE+5);
          ev_10 = sum(ev > BASE+6);
          ev_11 = sum(ev > BASE+7);
          ev_12 = sum(ev > BASE+8);
          ev_13 = sum(ev > BASE+9);
          ev_14 = sum(ev > BASE+10);
          ev_15 = sum(ev > BASE+11);
          
          mat_to_fill = [];
          if (mode == INTERICTAL_MODE)
            mat_to_fill = interict_mat;
          elseif (mode == PREICTAL_MODE) 
            mat_to_fill = preict_mat;
          endif 
          
          mat_to_fill(fi,1) = ev_5;
          mat_to_fill(fi,2) = ev_6;
          mat_to_fill(fi,3) = ev_7;
          mat_to_fill(fi,4) = ev_8;
          mat_to_fill(fi,5) = ev_9;
          mat_to_fill(fi,6) = ev_10;
          mat_to_fill(fi,7) = ev_11;
          mat_to_fill(fi,8) = ev_12;
          mat_to_fill(fi,9) = ev_13;
          mat_to_fill(fi,10) = ev_14;
          mat_to_fill(fi,11) = ev_15;
                   
          if (mode == INTERICTAL_MODE)
            interict_mat = mat_to_fill;
          elseif (mode == PREICTAL_MODE)
            preict_mat = mat_to_fill;
          endif
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      fflush(stdout);
    endfor 
  endfor 
endfor

TH=1;
mean(interict_mat)
std(interict_mat)
sum(interict_mat > TH) / size(interict_mat,1)
                   
mean(preict_mat)
std(preict_mat)
sum(preict_mat <= TH) / size(preict_mat,1)
                   




