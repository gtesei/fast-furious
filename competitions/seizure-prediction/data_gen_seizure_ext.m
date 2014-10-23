#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

########################### CONSTANTS 
INTERICTAL_MODE = 1;
PREICTAL_MODE = 2;
TEST_MODE = 3;

########################### loadind data sets and meta-data
printf("|--> generating features set ...\n");

%%dss = ["Dog_1"; "Dog_2"; "Dog_3"; "Dog_4"; "Dog_5"; "Patient_1"; "Patient_2"];
%%dss = ["Patient_1"; "Dog_1"; "Dog_2"];
dss = ["Patient_1"];
cdss = cellstr (dss);

printf("|--> found %i data sets ... \n",size(cdss,1));
for i = 1:size(cdss,1)
  printf("|- %i - %s \n",i,cell2mat(cdss(i,1)));
endfor 

for i = 1:size(cdss,1)
  ds = cell2mat(cdss(i,1)); 
  printf("|--> processing %s  ...\n",ds);
  
  %% making digest directory 
  dirname = [curr_dir "/dataset/seizure-prediction/" ds "_digest/"];
  mkdir(dirname); %% if the directory exists this doesn't do nothing 
  
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
  
  %%%%%%%%%% train / test matrix 
  tr_size = size(preictal_files,1)+size(interictal_files,1);
  ts_size = size(test_files,1);
  
  matrix_in = 0;
  Xtrain_mean_sd = [];
  Xtest_mean_sd = [];
  Xtrain_quant = [];
  Xtest_quant = [];
  ytrain = [];
  
  train_index = 1;
  %%%%%%%%%% main loop  
  for mode = 1:3 
    printf("|--> mode = %i (1 = inter,2=preict,3=test) ...\n",mode);
    
    files = test_files;
    if (mode == INTERICTAL_MODE)
      files = interictal_files;
    elseif (mode == PREICTAL_MODE) 
      files = preictal_files;
    endif 
    
    for fi = 1:size(files,1)
      fn = cell2mat(files(fi,1));
      printf("|- processing %s  ...\n",fn);
      
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
      
      %%% initializing matrices ... 
      if (matrix_in == 0) 
        Xtrain_mean_sd = zeros(tr_size,13*size(data,1));
        Xtest_mean_sd = zeros(ts_size,13*size(data,1));
        Xtrain_quant = zeros(tr_size,17*size(data,1));
        Xtest_quant = zeros(ts_size,17*size(data,1));
        ytrain = zeros(tr_size,2);
        matrix_in = 1;
      endif 
      
      name = cell2mat(seg_struct_names(2,1));
      data_length_sec = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(3,1));
      sampling_fraquency = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(4,1));
      channels = getfield(seg_struct,name);
      
      %%% time_before_seizure
      sequence = -1; 
      time_before_seizure = -1;
      if (mode != TEST_MODE) 
        name = cell2mat(seg_struct_names(5,1));
        sequence = getfield(seg_struct,name);
       
        if (mode == INTERICTAL_MODE)
          if (size(findstr(ds,"Dog"),1) > 0 ) 
            time_before_seizure = 7*24*60*60; %%% 1 week for interictal dogs 
          else 
            time_before_seizure = 4*60*60; %%% 4 hours for humans  
          endif 
        elseif (mode == PREICTAL_MODE)
          time_before_seizure = 60*60-sequence*60+60*5; %%% for preictal
        endif 
      
        %%% update ytrain  
        ytrain(train_index,1) = time_before_seizure;
        ytrain(train_index,2) = (mode-1);
      endif 
      
      %% garbage collection .. 
      clear seg_struct;
      
      %%% signal processing ... 
      for i = 1:size(data,1)
        sign = data(i,:)';
        
        %% fm and extended stuff 
        [fm] = findMainFrequencyComponent (sampling_fraquency,data_length_sec,sign,doPlot=0);
        [p1,p2,p3,p4,p5,P] = bandPower (sampling_fraquency,data_length_sec,sign,debug=0);
        [f_50] = findSpectralEdgeFrequency (sampling_fraquency,data_length_sec,sign);
        [min_tau] = findMinTimeAutocorrelationZero (sign,sampling_fraquency);
        skw = skewness (sign);
        kur = kurtosis (sign); 
        if (mode != TEST_MODE) 
          Xtrain_mean_sd(train_index,((i-1)*13+3)) = fm;
          Xtrain_mean_sd(train_index,((i-1)*13+4)) = P;
          Xtrain_mean_sd(train_index,((i-1)*13+5)) = p1;
          Xtrain_mean_sd(train_index,((i-1)*13+6)) = p2;
          Xtrain_mean_sd(train_index,((i-1)*13+7)) = p3;
          Xtrain_mean_sd(train_index,((i-1)*13+8)) = p4;
          Xtrain_mean_sd(train_index,((i-1)*13+9)) = p5; 
          Xtrain_mean_sd(train_index,((i-1)*13+10)) = f_50;
          Xtrain_mean_sd(train_index,((i-1)*13+11)) = min_tau;
          Xtrain_mean_sd(train_index,((i-1)*13+12)) = skw;
          Xtrain_mean_sd(train_index,((i-1)*13+13)) = kur;

          Xtrain_quant(train_index,((i-1)*17+7)) = fm;
          Xtrain_quant(train_index,((i-1)*17+8)) = P;
          Xtrain_quant(train_index,((i-1)*17+9)) = p1;
          Xtrain_quant(train_index,((i-1)*17+10)) = p2;
          Xtrain_quant(train_index,((i-1)*17+11)) = p3;
          Xtrain_quant(train_index,((i-1)*17+12)) = p4;
          Xtrain_quant(train_index,((i-1)*17+13)) = p5;
          Xtrain_quant(train_index,((i-1)*17+14)) = f_50;
          Xtrain_quant(train_index,((i-1)*17+15)) = min_tau;
          Xtrain_quant(train_index,((i-1)*17+16)) = skw;
          Xtrain_quant(train_index,((i-1)*17+17)) = kur;
        else 
          Xtest_mean_sd(fi,((i-1)*13+3)) = fm;
          Xtest_mean_sd(fi,((i-1)*13+4)) = P;
          Xtest_mean_sd(fi,((i-1)*13+5)) = p1;
          Xtest_mean_sd(fi,((i-1)*13+6)) = p2;
          Xtest_mean_sd(fi,((i-1)*13+7)) = p3;
          Xtest_mean_sd(fi,((i-1)*13+8)) = p4;
          Xtest_mean_sd(fi,((i-1)*13+9)) = p5;
          Xtest_mean_sd(fi,((i-1)*13+10)) = f_50;
          Xtest_mean_sd(fi,((i-1)*13+11)) = min_tau;
          Xtest_mean_sd(fi,((i-1)*13+12)) = skw;
          Xtest_mean_sd(fi,((i-1)*13+13)) = kur;

          Xtest_quant(fi,((i-1)*17+7)) = fm;
          Xtest_quant(fi,((i-1)*17+8)) = P;
          Xtest_quant(fi,((i-1)*17+9)) = p1;
          Xtest_quant(fi,((i-1)*17+10)) = p2;
          Xtest_quant(fi,((i-1)*17+11)) = p3;
          Xtest_quant(fi,((i-1)*17+12)) = p4;
          Xtest_quant(fi,((i-1)*17+13)) = p5;
          Xtest_quant(fi,((i-1)*17+14)) = f_50;
          Xtest_quant(fi,((i-1)*17+15)) = min_tau;
          Xtest_quant(fi,((i-1)*17+16)) = skw;
          Xtest_quant(fi,((i-1)*17+17)) = kur;
        endif 
        
        %% mu, sd 
        mu = mean(sign);
        sd = std(sign);
        if (mode != TEST_MODE) 
          Xtrain_mean_sd(train_index,((i-1)*13+1)) = mu;
          Xtrain_mean_sd(train_index,((i-1)*13+2)) = sd;
        else 
          Xtest_mean_sd(fi,((i-1)*13+1)) = mu;
          Xtest_mean_sd(fi,((i-1)*13+2)) = sd;
        endif 
        
        %% quantiles 
        q = quantile (sign, [0.05 0.15 0.35 0.5 0.65 0.85]);
        if (mode != TEST_MODE) 
	  Xtrain_quant( train_index , ((i-1)*17+1):((i-1)*17+6) ) = q; 
	else 
	  Xtest_quant( fi , ((i-1)*17+1):((i-1)*17+6) ) = q;        
        endif 
      endfor 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      if (mode != TEST_MODE)
        train_index = (train_index + 1);
      endif 
      
      fflush(stdout);
    endfor 
  endfor 
  
  %%%%%%%%%%%%% serializing matrices
  dlmwrite([dirname "Xtrain_mean_sd.zat"] , Xtrain_mean_sd);
  dlmwrite([dirname "Xtest_mean_sd.zat"]  , Xtest_mean_sd);
 
  dlmwrite([dirname "Xtrain_quant.zat"]   , Xtrain_quant);
  dlmwrite([dirname "Xtest_quant.zat"]    , Xtest_quant);
  
  dlmwrite([dirname "ytrain.zat"] , ytrain);
endfor 
