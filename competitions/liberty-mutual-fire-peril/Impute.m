#! /opt/local/bin/octave -qf 

%%%% making enviroment 
menv;

########################### loadind data sets and meta-data
printf("|--> loading meta-data ...\n");
ImputePredictors = dlmread([curr_dir "/dataset/liberty-mutual-fire-peril/ImputePredictors4octave.zat"]);


## elimina le intestazioni del csv  
ImputePredictors = ImputePredictors(2:end,:);

########################### pocessing meta-data and finding best models .. TODO
pNum = size(ImputePredictors,1)
for i = 1:pNum
  if (ImputePredictors(i,3))

  endif
endfor



