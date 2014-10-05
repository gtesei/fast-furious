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
dss = ["Patient_2"; "Dog_3"; "Dog_4"; "Dog_5"];
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
  mkdir(dirname);
  
  %% data files 
  pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_interictal_segment*"]);
  interictal_files = glob (pattern_interictal);
  
  pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_preictal_segment*"]);
  preictal_files = glob (pattern_preictal);
  
  pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_test_segment*"]);
  test_files = glob (pattern_test);
  
  %%%%%%%%%% train / test matrix 
  tr_size = size(preictal_files,1)+size(interictal_files,1);
  ts_size = size(test_files,1);
  
  Xtrain_mean_sd = zeros(tr_size,3*16);
  Xtest_mean_sd = zeros(ts_size,3*16);
  
  Xtrain_quant = zeros(tr_size,7*16);
  Xtest_quant = zeros(ts_size,7*16);
  
  ytrain = zeros(tr_size,2);
  
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
      
      seg_struct_names = fieldnames(seg_struct);
      
      name = cell2mat(seg_struct_names(1,1));
      data = getfield(seg_struct,name);
      
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
      
      %%% signal processing ...
      eNum = 16;
      if (size(findstr(ds,"Patient"),1) > 0 ) 
        eNum = 15;
      endif  
      for i = 1:eNum
        sign = data(i,:)';
        
        %% fm 
        [fm] = findMainFrequencyComponent (sampling_fraquency,data_length_sec,sign,doPlot=0);
        if (mode != TEST_MODE) 
          Xtrain_mean_sd(train_index,((i-1)*3+3)) = fm;
          Xtrain_quant(train_index,((i-1)*7+7)) = fm;
        else 
          Xtest_mean_sd(fi,((i-1)*3+3)) = fm;
          Xtest_quant(fi,((i-1)*7+7)) = fm;
        endif 
        
        %% mu, sd 
        mu = mean(sign);
        sd = std(sign);
        if (mode != TEST_MODE) 
          Xtrain_mean_sd(train_index,((i-1)*3+1)) = mu;
          Xtrain_mean_sd(train_index,((i-1)*3+2)) = sd;
        else 
          Xtest_mean_sd(fi,((i-1)*3+1)) = mu;
          Xtest_mean_sd(fi,((i-1)*3+2)) = sd;
        endif 
        
        %% quantiles 
        q = quantile (sign, [0.05 0.15 0.35 0.5 0.65 0.85]);
        if (mode != TEST_MODE) 
	  Xtrain_quant( train_index , ((i-1)*7+1):((i-1)*7+6) ) = q; 
	else 
	  Xtest_quant( fi , ((i-1)*7+1):((i-1)*7+6) ) = q;        
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