#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

########################### CONSTANTS 
INTERICTAL_MODE = 1;
PREICTAL_MODE = 2;
TEST_MODE = 3;

###########################  loadind data sets and meta-data
printf("|--> generating features set ...\n");

%%dss = ["Dog_1"; "Dog_2"; "Dog_3"; "Dog_4"; "Dog_5"; "Patient_1"; "Patient_2"];
dss = ["Dog_2"; "Dog_4"; "Dog_5"; "Patient_1"];
cdss = cellstr (dss);

printf("|--> found %i data sets ... \n",size(cdss,1));
for i = 1:size(cdss,1)
  printf("|- %i - %s \n",i,cell2mat(cdss(i,1)));
endfor 

for i = 1:size(cdss,1)
  ds = cell2mat(cdss(i,1)); 
  printf("|--> processing %s  ...\n",ds);
  
  %% making digest directory 
  dirname = [curr_dir "/dataset/seizure-prediction/" ds "_feature_pack/"];
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
  Xtrain_pack = [];
  Xtest_pack = [];
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
      
      name = cell2mat(seg_struct_names(2,1));
      data_length_sec = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(3,1));
      sampling_fraquency = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(4,1));
      channels = getfield(seg_struct,name);
      
      %%% initializing matrices ... 
      if (matrix_in == 0) 
        ncol = 4 * length(channels);
       
        Xtrain_pack = zeros(tr_size,ncol);
        Xtest_pack = zeros(ts_size,ncol);
        
        ytrain = zeros(tr_size,1);
        
        matrix_in = 1;
      endif 
      
      %% garbage collection .. 
      clear seg_struct;
      
      if (mode != TEST_MODE) 
        ytrain(train_index,1) = (mode-1);
        printf("|- classified as  %i  (0=inter,1=preict) - train_index = %i...\n",ytrain(train_index,1) , train_index );
      endif
      
      %%% signal processing ... 
      for i = 1:size(data,1)
        sign = data(i,:)';    
        sign2 = sign .^ 2;
        
        mu = mean(sign2);
        sd = std(sign2);
        skw = skewness (sign2);
        kur = kurtosis (sign2); 
        ##etot = sum(sign2);
        
        ##[LLE LLE_mean LLE_sd]=lyaprosen(sign,0,0);
        
        if (mode != TEST_MODE)
          Xtrain_pack(train_index,(i-1)*4+1) = mu;
          Xtrain_pack(train_index,(i-1)*4+2) = sd;
          Xtrain_pack(train_index,(i-1)*4+3) = skw;
          Xtrain_pack(train_index,(i-1)*4+4) = kur;
          ##Xtrain_pack(train_index,(i-1)*5+5) = etot;
          ##Xtrain_pack(train_index,(i-1)*8+6) = LLE;
          ##Xtrain_pack(train_index,(i-1)*8+7) = LLE_mean;
          ##Xtrain_pack(train_index,(i-1)*8+8) = LLE_sd;
        else 
          Xtest_pack(fi,(i-1)*4+1) = mu;
          Xtest_pack(fi,(i-1)*4+2) = sd;
          Xtest_pack(fi,(i-1)*4+3) = skw;
          Xtest_pack(fi,(i-1)*4+4) = kur;
          ##Xtest_pack(fi,(i-1)*5+5) = etot;
          ##Xtest_pack(fi,(i-1)*8+6) = LLE;
          ##Xtest_pack(fi,(i-1)*8+7) = LLE_mean;
          ##Xtest_pack(fi,(i-1)*8+8) = LLE_sd;
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
  dlmwrite([dirname "Xtrain_pack.zat"] , Xtrain_pack);
  dlmwrite([dirname "Xtest_pack.zat"]  , Xtest_pack);

endfor
