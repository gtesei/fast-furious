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
  dirname = [curr_dir "/dataset/seizure-prediction/" ds "_pca_feateare/"];
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
  Xtrain_pca = [];
  Xtest_pca = [];
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
        ncol = 0;
        if (strcmp(ds,"Dog_5")) 
          ncol =1;
        elseif (strcmp(ds,"Patient_2"))
          ncol=4;
        elseif (strcmp(ds,"Patient_1"))
          ncol=4;
        else 
         error("for this data set the relation doen't hold."); 
        endif 
        Xtrain_pca = zeros(tr_size,ncol);
        Xtest_pca = zeros(ts_size,ncol);
        
        ytrain = zeros(tr_size,1);
        
        matrix_in = 1;
      endif 
      
      name = cell2mat(seg_struct_names(2,1));
      data_length_sec = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(3,1));
      sampling_fraquency = getfield(seg_struct,name);
      
      name = cell2mat(seg_struct_names(4,1));
      channels = getfield(seg_struct,name);
      
      %% garbage collection .. 
      clear seg_struct;
      
      
      if (mode != TEST_MODE) 
        ytrain(train_index,1) = (mode-1);
        printf("|- classified as  %i  (0=inter,1=preict) - train_index = %i...\n",ytrain(train_index,1) , train_index );
      endif
      
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
          
          if (strcmp(ds,"Dog_5")) 
            ev_13 = sum(ev > 13);
            if (mode != TEST_MODE) 
              Xtrain_pca(train_index,1) = ev_13;
            else 
              Xtest_pca(fi,1)  = ev_13; 
            endif 
          elseif (strcmp(ds,"Patient_2") | strcmp(ds,"Patient_1"))
            ev_8 = sum(ev > 8);
            ev_9 = sum(ev > 9);
            ev_10 = sum(ev > 10);
            ev_11 = sum(ev > 11);
            if (mode != TEST_MODE) 
              Xtrain_pca(train_index,1) = ev_8;
              Xtrain_pca(train_index,2) = ev_9;
              Xtrain_pca(train_index,3) = ev_10;
              Xtrain_pca(train_index,4) = ev_11;
            else 
              Xtest_pca(fi,1)  = ev_8; 
              Xtest_pca(fi,2)  = ev_9; 
              Xtest_pca(fi,3)  = ev_10; 
              Xtest_pca(fi,4)  = ev_11; 
            endif
          else 
           error("for this data set the relation doen't hold."); 
          endif 
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      if (mode != TEST_MODE)
        train_index = (train_index + 1);
      endif 
      
      fflush(stdout);
    endfor 
  endfor 
  
  %%%%%%%%%%%%% serializing matrices
  dlmwrite([dirname "Xtrain_pca.zat"] , Xtrain_pca);
  dlmwrite([dirname "Xtest_pca.zat"]  , Xtest_pca);
  
  %%%%%%%% check 
  printf("|-- checking interict \n")
  sum( Xtrain_pca(ytrain == 0 ,:) > 1) / sum(ytrain == 0)
  printf("|-- checking preict \n")
  sum( Xtrain_pca(ytrain == 1 ,:) <= 1) / sum(ytrain == 1)
endfor 