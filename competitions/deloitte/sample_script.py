# Load required libraries and datasets

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

path = ""
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# One-hot-encode categorical variables
train['dataset'] = "train"
test['dataset'] = "test"
data = pd.concat([train,test], axis = 0)
categorical = ['property_type','room_type','bed_type','cancellation_policy','city']
data = pd.get_dummies(data, columns = categorical)



# Select only numeric data and impute missing values as 0
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_x = data[data.dataset == "train"] \
    .select_dtypes(include=numerics) \
    .drop("log_price", axis = 1) \
    .fillna(0) \
    .values

test_x = data[data.dataset == "test"] \
    .select_dtypes(include=numerics) \
    .drop("log_price", axis = 1) \
    .fillna(0) \
    .values
    
train_y = data[data.dataset == "train"].log_price.values

# Train a Random Forest model with cross-validation

from sklearn.model_selection import KFold
cv_groups = KFold(n_splits=3)
regr = RandomForestRegressor(random_state = 0, n_estimators = 10)

for train_index, test_index in cv_groups.split(train_x):
    
    # Train the model using the training sets
    regr.fit(train_x[train_index], train_y[train_index])
    
    # Make predictions using the testing set
    pred_rf = regr.predict(train_x[test_index])
    
    # Calculate RMSE for current cross-validation split
    rmse = str(np.sqrt(np.mean((train_y[test_index] - pred_rf)**2)))
    
    print("RMSE for current split: " + rmse)
    
# Create submission file
regr.fit(train_x, train_y)
final_prediction = regr.predict(test_x)

submission = pd.DataFrame(np.column_stack([test.id, final_prediction]), columns = ['id','log_price'])
submission.to_csv("sample_submission.csv", index = False)
