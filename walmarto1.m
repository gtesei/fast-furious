#! /opt/local/bin/octave -qf 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% setting enviroment 
menv;

test.all = csvread ([curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/test.zat"]);
sampleSubmission = csvread ([curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/sampleSubmission.zat"]);


### 1 load test.csv (to get ids) and samplesubmission to create hash map for predictions 

### 2 loop in id in ids, load train.csv , test.csv , train , prediction 

### 3 store prediction in hash map deepid <--> pred 

### build submision file 
# aa = getfield (hash , '1_1_2022') 
# a = ["1_1_2022" "," num2str(aa)]
# ss = [a ; a ; a ; a ; ]
# dlmwrite("aa.csv",ss , "delimiter", "")
