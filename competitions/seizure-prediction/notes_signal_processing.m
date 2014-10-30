#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

########################### CONSTANTS 
INTERICTAL_MODE = 1;
PREICTAL_MODE = 2;
TEST_MODE = 3;

NUMBER_OF_FILES = 1;
NUMBER_CHANNEL_PER_FILE = 2;

########################### loadind data sets and meta-data
printf("|--> generating features set ...\n");

%%dss = ["Dog_1"; "Dog_2"; "Dog_3"; "Dog_4"; "Dog_5"; "Patient_1"; "Patient_2"];
dss = ["Dog_1" ; "Dog_3"; "Patient_1"];
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
      for i = 1:size(data,1)
        if (i > NUMBER_CHANNEL_PER_FILE)
          break; 
        endif 
        printf("|- processing channel n. %i  ...\n",i);
        
        sign = data(i,:)';
     
        %%%%%%
        t = 0:1/sampling_fraquency:data_length_sec-1/sampling_fraquency;      
        n = length(t);
        printf("|- signal with %i tics in time domain - T = %i sec. \n",n,data_length_sec);
  
        swaveX = fft(sign)/ n;
        hz = linspace( 0, sampling_fraquency/2, floor(n/2) + 1);
  
        nyquistfreq = sampling_fraquency/2;
        
        printf("|- signal with %i tics in frequency domain - freq. nyquist = %f  \n",length(hz),nyquistfreq);
        
        ### (1) la potenza del segnale e' quasi tutta entro i 40 hz?
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
        
        printf("|- power signal = %f  \n", P );
        printf("|- p0 (<0.5 hz) = %f  \n", p0 );
        printf("|- p1 ( 0.5 hz < sign < 4 hz) = %f  \n", p1 );
        printf("|- p2 ( 4 hz < sign < 8 hz) = %f  \n", p2 );
        printf("|- p3 ( 8 hz < sign  < 13 hz) = %f  \n", p3 );
        printf("|- p4 ( 13 hz < sign < 30 hz) = %f  \n", p4 );
        printf("|- p5 ( 30 hz < sign < 48 hz) = %f  \n", p5 );
        printf("|- p6 ( 48 hz < sign < 90 hz) = %f  \n", p6 );
        printf("|- p7 ( 90 hz < sign < 150 hz) = %f  \n", p7 );
        printf("|- pTail ( 150 hz < sign < nyquist hz) = %f  \n", pTail );
        
        [fm] = findMainFrequencyComponent (sampling_fraquency,data_length_sec,sign,doPlot=0)
        [p0,p1,p2,p3,p4,p5,p6,pTail,P] = bandPower (sampling_fraquency,data_length_sec,sign,debug=0,norm=1)

        pause;
        
        
        ### (2) c'e' davvero cosi tanto rumore da dover smothare il segnale con mov avg 12? 
        #d = 6; % 12-point mean filter 
        #dataMean = zeros(size( sign)); 
        #dataMed = zeros(size( sign)); 
        #for i = d + 1: length( t)-d-1 
        #  dataMean(i) = mean( sign( i-d:i + d)); 
        #  dataMed(i) = median( data( i-d:i + d));
        #end
        
        wndw = 12;                                      %# sliding window size
        dataMean = filter(ones(wndw,1)/wndw, 1, sign); %# moving average
        
        tt = 1:(sampling_fraquency*5);  
        subplot (2, 1, 1);
        plot(tt,sign(tt));
        title('signal');
        subplot (2, 1, 2);
        plot(tt,dataMean(tt));
        title('mean 12');
        pause;
        
        
        subplot( 211) 
        plot( hz, 2* abs( swaveX( 1: length( hz)))), hold on 
        plot( hz, 2* abs( swaveX( 1: length( hz))),' r') 
        xlabel(' Frequencies (Hz)') 
        ylabel(' Amplitude') 
        legend({'fast Fourier transform'}) 
        
        subplot( 212) 
        plot( hz, angle( swaveX( 1: floor( n/ 2) + 1))) 
        xlabel(' Frequencies (Hz)') 
        ylabel(' Phase (radians)')
        
        %%%%%%
        t = 0:1/sampling_fraquency:data_length_sec-1/sampling_fraquency;      
        n = length(t);
        printf("|- smoothed - signal smoothed with %i tics in time domain - T = %i sec. \n",n,data_length_sec);
  
        swaveX = fft(dataMean)/ n;
        hz = linspace( 0, sampling_fraquency/2, floor(n/2) + 1);
  
        nyquistfreq = sampling_fraquency/2;
        
        printf("|- smoothed - signal with %i tics in frequency domain - freq. nyquist = %f  \n",length(hz),nyquistfreq);
       
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
        
        printf("|- smoothed - power signal = %f  \n", P );
        printf("|- smoothed - p0 (<0.5 hz) = %f  \n", p0 );
        printf("|- smoothed - p1 ( 0.5 hz < sign < 4 hz) = %f  \n", p1 );
        printf("|- smoothed - p2 ( 4 hz < sign < 8 hz) = %f  \n", p2 );
        printf("|- smoothed - p3 ( 8 hz < sign  < 13 hz) = %f  \n", p3 );
        printf("|- smoothed - p4 ( 13 hz < sign < 30 hz) = %f  \n", p4 );
        printf("|- smoothed - p5 ( 30 hz < sign < 48 hz) = %f  \n", p5 );
        printf("|- smoothed - p6 ( 48 hz < sign < 90 hz) = %f  \n", p6 );
        printf("|- smoothed - p7 ( 90 hz < sign < 150 hz) = %f  \n", p7 );
        printf("|- smoothed - pTail ( 150 hz < sign < nyquist hz) = %f  \n", pTail );
        
        [fm] = findMainFrequencyComponent (sampling_fraquency,data_length_sec,sign,doPlot=0)
    
        pause;
        
        
        endfor 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      fflush(stdout);
    endfor 
  endfor 
endfor 
    
