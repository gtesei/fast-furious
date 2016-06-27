# coding: utf-8
__author__ = 'Sandro Vega Pons : https://www.kaggle.com/svpons'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import xgboost as xgb


def prepare_data(df_train, df_test, n_cell_x, n_cell_y):
    """
    Some feature engineering (mainly with the time feature) + normalization
    of all features (substracting the mean and dividing by std) +
    computation of a grid (size = n_cell_x * n_cell_y), which is included
    as a new column (grid_cell) in the dataframes.

    Parameters:
    ----------
    df_train: pandas DataFrame
              Training data
    df_test : pandas DataFrame
              Test data
    n_cell_x: int
              Number of grid cells on the x axis
    n_cell_y: int
              Number of grid cells on the y axis

    Returns:
    -------
    df_train, df_test: pandas DataFrame
                       Modified training and test datasets.
    """
    print('Feature engineering...')
    print('    Computing some features from x and y ...')
    ##x, y, and accuracy remain the same
    ##New feature x/y
    eps = 0.00001  # required to avoid some divisions by zero.
    df_train['x_d_y'] = df_train.x.values / (df_train.y.values + eps)
    df_test['x_d_y'] = df_test.x.values / (df_test.y.values + eps)
    ##New feature x*y
    df_train['x_t_y'] = df_train.x.values * df_train.y.values
    df_test['x_t_y'] = df_test.x.values * df_test.y.values

    print('    Creating datetime features ...')
    ##time related features (assuming the time = minutes)
    initial_date = np.datetime64('2014-01-01T01:01',  # Arbitrary decision
                                 dtype='datetime64[m]')
    # working on df_train
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                               for mn in df_train.time.values)
    df_train['hour'] = d_times.hour
    df_train['weekday'] = d_times.weekday
    df_train['day'] = d_times.day
    df_train['month'] = d_times.month
    df_train['year'] = d_times.year
    df_train = df_train.drop(['time'], axis=1)
    # working on df_test
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                               for mn in df_test.time.values)
    df_test['hour'] = d_times.hour
    df_test['weekday'] = d_times.weekday
    df_test['day'] = d_times.day
    df_test['month'] = d_times.month
    df_test['year'] = d_times.year
    df_test = df_test.drop(['time'], axis=1)

    print('Computing the grid ...')
    # Creating a new colum with grid_cell id  (there will be
    # n = (n_cell_x * n_cell_y) cells enumerated from 0 to n-1)
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    # df_train
    xs = np.where(df_train.x.values < eps, 0, df_train.x.values - eps)
    ys = np.where(df_train.y.values < eps, 0, df_train.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df_train['grid_cell'] = pos_y * n_cell_x + pos_x
    # df_test
    xs = np.where(df_test.x.values < eps, 0, df_test.x.values - eps)
    ys = np.where(df_test.y.values < eps, 0, df_test.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df_test['grid_cell'] = pos_y * n_cell_x + pos_x

    ##Normalization
    # print('Normalizing the data: (X - mean(X)) / std(X) ...')
    # cols = ['x', 'y', 'accuracy', 'x_d_y', 'x_t_y', 'hour',
    #         'weekday', 'day', 'month', 'year']
    # for cl in cols:
    #     ave = df_train[cl].mean()
    #     std = df_train[cl].std()
    #     df_train[cl] = (df_train[cl].values - ave) / std
    #     df_test[cl] = (df_test[cl].values - ave) / std

    # Returning the modified dataframes
    return df_train, df_test


def process_one_cell(df_train, df_test, grid_id, th):
    """
    Does all the processing inside a single grid cell: Computes the training
    and test sets inside the cell. Fits a classifier to the training data
    and predicts on the test data. Selects the top 3 predictions.

    Parameters:
    ----------
    df_train: pandas DataFrame
              Training set
    df_test: pandas DataFrame
             Test set
    grid_id: int
             The id of the grid to be analyzed
    th: int
       Threshold for place_id. Only samples with place_id with at least th
       occurrences are kept in the training set.

    Return:
    ------
    pred_labels: numpy ndarray
                 Array with the prediction of the top 3 labels for each sample
    row_ids: IDs of the samples in the submission dataframe
    """
    # Working on df_train
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    place_counts = df_cell_train.place_id.value_counts()
    mask = place_counts[df_cell_train.place_id.values] >= th
    df_cell_train = df_cell_train.loc[mask.values]

    # Working on df_test
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    row_ids = df_cell_test.index

    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'grid_cell'], axis=1).values
    X_test = df_cell_test.drop(['grid_cell'], axis=1).values
    if (X_test.shape[0] > 0):
        # Training Classifier
        if (X.shape[0] == 0):
            print("empty training set - grid_id:"+str(grid_id))


        #############
        vp = 0.1
        round(X.shape[0] * (1-vp))
        X_train = X[0:round(X.shape[0] * (1-vp))]
        y_train = y[0:round(X.shape[0] * (1-vp))]

        X_val = X[round(X.shape[0] * (1-vp)):]
        y_val = y[round(X.shape[0] * (1-vp)):]

        ###
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_val = xgb.DMatrix(X_val, label=y_val)
        xg_test = xgb.DMatrix(X_test)

        watchlist = [(xg_train, 'train'), (xg_val, 'eval')]

        ####
        param = {}
        # use softmax multi-class classification
        # param['objective'] = 'multi:softmax'
        param['objective'] = 'multi:softprob'
        param["eta"] = 0.1
        param["min_child_weight"] = 10
        param["subsample"] = 0.8
        param["colsample_bytree"] = 0.8
        param["scale_pos_weight"] = 1.0
        param["silent"] = 1
        param["max_depth"] = 7
        param["nthread"] = 4
        param['num_class'] = le.classes_.shape[0]
        param['eval_metric'] = 'merror'

        params = list(param.items())

        ###
        num_rounds = 30000
        model = xgb.train(params,
                          xg_train,
                          num_rounds,
                          watchlist,
                          # feval=mapk,
                          early_stopping_rounds=10)

        y_pred = model.predict(xg_test)
        #############

        pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:, ::-1][:, :3])
        return pred_labels, row_ids
    else:
        print("X_test.shape == 0 ... ")
        return [], row_ids


def process_grid(df_train, df_test, df_sub, th, n_cells):
    """
    Iterates over all grid cells and aggregates the results of individual cells
    """
    for g_id in range(n_cells):
        if g_id % 10 == 0:
            print('iteration: %s' % (g_id))

        # Applying classifier to one grid cell
        pred_labels, row_ids = process_one_cell(df_train, df_test, g_id, th)
        # Converting the prediction to the submission format
        str_labels = np.apply_along_axis(lambda x: ' '.join(x.astype(str)),
                                         1, pred_labels)
        # Updating submission file
        df_sub.loc[row_ids] = str_labels.reshape(-1, 1)

    return df_sub


if __name__ == '__main__':
    print('Loading data ...')
    df_train = pd.read_csv('/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/facebook-v-predicting-check-ins/train.csv', dtype={
                                                        'x': np.float32,
                                                        'y': np.float32,
                                                        'accuracy': np.int16,
                                                        'time': np.int64,
                                                        'place_id': np.int64},
                           index_col=0)
    df_test = pd.read_csv('/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/facebook-v-predicting-check-ins/test.csv', dtype={
                                                      'x': np.float32,
                                                      'y': np.float32,
                                                      'accuracy': np.int16,
                                                      'time': np.int64},
                          index_col=0)
    df_sub = pd.read_csv('/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/facebook-v-predicting-check-ins/sample_submission.csv', index_col=0)

    # Defining the size of the grid
    n_cell_x = 20  #10
    n_cell_y = 40  #30
    df_train, df_test = prepare_data(df_train, df_test, n_cell_x, n_cell_y)

    # Solving classification problems inside each grid cell
    th = 5 # 3 - Threshold on place_id inside each cell. Only place_ids with at
    # least th occurrences inside each grid_cell are considered. This
    # is to avoid classes with very few samples and speed-up the
    # computation.

    df_submission = process_grid(df_train, df_test, df_sub, th,
                                 n_cell_x * n_cell_y)
    # creating the submission
    print('Generating submission file ...')
    df_submission.to_csv("/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/facebook-v-predicting-check-ins/sub_xgb.csv", index=True)
