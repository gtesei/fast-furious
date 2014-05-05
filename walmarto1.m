#! /opt/local/bin/octave -qf 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% setting enviroment 
menv;

### 1 load ids.csv and samplesubmission to create hash map for predictions 
ids = dlmread ([curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/ids.csv"]); #	Store, Dept 

### 2 loop in id in ids, load train.csv , test.csv , train , prediction 
ids_num = size(ids)(1,1);
for i = 1:ids_num 
	store = ids(i,1);
	dept = ids(i,2);
	
	## load train set 
	##Store,Dept,Weekly_Sales,store.type,store.size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment
	train_fn = [curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/gen_oct/" num2str(store) "_" num2str(dept) "_train.zat"];
	train = dlmread (train_fn);
	
	## load test set
	## Store,Dept,store.type,store.size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,IsHoliday
	test_fn = [curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/gen_oct/" num2str(store) "_" num2str(dept) "_test.zat"];
	test = dlmread (test_fn);
	
	## training and xval 
	
	
	
	## predicting 
	
endfor 


### 3 store prediction in hash map deepid <--> pred 

### build submision file 
# aa = getfield (hash , '1_1_2022') 
# a = ["1_1_2022" "," num2str(aa)]
# ss = [a ; a ; a ; a ; ]
# dlmwrite("aa.csv",ss , "delimiter", "")
