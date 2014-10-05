#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

########################### loadind data sets and meta-data
printf("|--> loading meta-data ...\n");

ds = "Dog_1";

pattern_interictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_interictal_segment*"]);
interictal_files = glob (pattern_interictal);

pattern_preictal = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_preictal_segment*"]);
preictal_files = glob (pattern_preictal);

pattern_test = ([curr_dir "/dataset/seizure-prediction/" ds "/" ds "/" ds "_test_segment*"]);
test_files = glob (pattern_test);

afile = cell2mat(test_files(1,1)); %% es. il path del primo file ..  
nfiles = size(test_files,1);

%seg = load  ([curr_dir "/dataset/seizure-prediction/Dog_1/Dog_1/Dog_1_interictal_segment_0015.mat"]);
seg = load  ([curr_dir "/dataset/seizure-prediction/Dog_1/Dog_1/Dog_1_preictal_segment_0021.mat"]);

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

name = cell2mat(seg_struct_names(5,1));
sequence = getfield(seg_struct,name);

time_before_seizere = 7*24*60*60; %%% 1 week for interictal dogs 
time_before_seizere = 60*60-sequence*60+60*5; %%% for preictal

for i = 1:16
  sign = data(i,:)';
  [fm] = findMainFrequencyComponent (sampling_fraquency,data_length_sec,sign,doPlot=0);
  mu = mean(sign);
  sd = std(sign);
  printf("electrodote n. %i - fm=%f , mu=%f , sd=%f  \n",i,fm,mu,sd);
  q = quantile (sign, [0.15 0.30 0.5 0.60 0.75]) 
endfor 










