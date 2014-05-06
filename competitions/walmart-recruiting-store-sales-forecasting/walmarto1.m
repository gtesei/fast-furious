#! /opt/local/bin/octave -qf 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% setting enviroment 
menv;

### 1 load ids.csv and samplesubmission to create hash map for predictions 
ids = dlmread ([curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/ids.csv"]); #	Store, Dept 

### 2 loop in id in ids, load train.csv , test.csv , train , prediction 
ids_num = size(ids)(1,1);
#ids_num = 10
for i = 1:ids_num 
	store = ids(i,1);
	dept = ids(i,2);
	
	## load train set, findind best p, lambda ...  
	##Store,Dept,Weekly_Sales,store.type,store.size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,IsHoliday
	train_fn = [curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/gen_oct/" num2str(store) "_" num2str(dept) "_train.zat"];
        
        train = zeros(1,1);

        try
         train = dlmread (train_fn);
       catch
         ;
       end_try_catch


	
        if (size(train)(1,1) < 10) 
          continue;
        endif  

        y = train(:,3);
        X = train(:,4:end); 	
        [Xtrain,ytrain,Xval,yval] = splitTrainValidation(X,y,0.70);

        printf("|--> finding optimal polinomial degree ... \n");
        [p_opt,J_opt] = findOptP_RegLin(Xtrain, ytrain, Xval, yval); 
        printf("|--> found p_opt == %i \n",p_opt)

        printf("|--> finding optimal regularization parameter ... \n");
        [lambda_opt,J_opt] = findOptLambda_RegLin(Xtrain, ytrain, Xval, yval); 
        printf("|--> found lambda_opt == %i \n",lambda_opt)

        #printf("|--> computing learning curve ... \n");
        #tic(); [error_train,error_val] = learningCurve_RegLin(Xtrain, ytrain, Xval, yval); toc();        

        ## training with best p, lambda 
        [X_norm,mu,sigma] = treatContFeatures(X,p_opt);
        [theta] = trainLinearReg(X_norm, y, lambda_opt , 400 );  

	## load test set
	## Store,Dept,store.type,store.size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,IsHoliday
	test_fn = [curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/gen_oct/" num2str(store) "_" num2str(dept) "_test.zat"];
	test = dlmread (test_fn);
	
	## predicting 
        Xtest = test(:,3:end);
        [Xtest_norm,mu_val,sigma_val] = treatContFeatures(Xtest,p_opt,1,mu,sigma);
        if (mu != mu_val | sigma != sigma_val) 
          disp(mu); disp(mu_val); disp(sigma); disp(sigma_val);
          error("error in function treatContFeatures: mu != mu_val or sigma != sigma_val - displayed mu,mu_val,sigma,sigma_val .. ");
        endif
        y_pred = predictLinearReg(Xtest_norm,theta);
        
        dlmwrite ([curr_dir "/dataset/walmart-recruiting-store-sales-forecasting/gen_oct/" num2str(store) "_" num2str(dept) "_pred.zat"] , y_pred);
endfor 


### 3 store prediction in hash map deepid <--> pred 

### build submision file 
# aa = getfield (hash , '1_1_2022') 
# a = ["1_1_2022" "," num2str(aa)]
# ss = [a ; a ; a ; a ; ]
# dlmwrite("aa.csv",ss , "delimiter", "")
