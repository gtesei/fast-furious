
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import itertools

def prepare_data(df, n_cell_x, n_cell_y):
    """
    Feature engineering and computation of the grid.
    """
    # Creating the grid
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    eps = 0.00001
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df['grid_cell'] = pos_y * n_cell_x + pos_x

    return df
if __name__ == '__main__':
    print('Loading data ...')
    df_train = pd.read_csv('C:\\Users\\gtesei\\Desktop\\Deloitte\\C_Folder\\Cognitive_Technologies\\Machine_Learning\\git\\fast-furious\dataset\\facebook-v-predicting-check-ins\\train.csv', dtype={
                                                        'x': np.float32,
                                                        'y': np.float32,
                                                        'accuracy': np.int16,
                                                        'time': np.int64,
                                                        'place_id': np.int64},
                           index_col=0)
    df_test = pd.read_csv('C:\\Users\\gtesei\\Desktop\\Deloitte\\C_Folder\\Cognitive_Technologies\\Machine_Learning\\git\\fast-furious\dataset\\facebook-v-predicting-check-ins\\test.csv', dtype={
                                                      'x': np.float32,
                                                      'y': np.float32,
                                                      'accuracy': np.int16,
                                                      'time': np.int64},
                          index_col=0)
    df_sub = pd.read_csv('C:\\Users\\gtesei\\Desktop\\Deloitte\\C_Folder\\Cognitive_Technologies\\Machine_Learning\\git\\fast-furious\dataset\\facebook-v-predicting-check-ins\\sample_submission.csv', index_col=0)

    df_train = df_train.drop(['accuracy'], 1)
    df_test = df_test.drop(['accuracy'], 1)

    n_cell_x = 10
    n_cell_y = 30

    assert df_train[df_train.place_id.values < 0].shape[0] == 0

    print('Preparing train data')
    df_train = prepare_data(df_train, n_cell_x, n_cell_y)

    print('Preparing test data')
    df_test = prepare_data(df_test, n_cell_x, n_cell_y)

    ##
    grid_id = 180
    th = 2
    g_dict = {'delta_time':[60, 3 * 60, 24 * 40, 2 * 24 * 60, 4 * 24 * 60, 7 * 24 * 60, 15 * 24 * 60, 30 * 24 * 60, 45 * 24 * 60],
                             'k':[5,10,15,20,25,30,35]}
    cv_grid = pd.DataFrame(list(itertools.product(*g_dict.values())), columns=g_dict.keys())
    cv_grid['MAP@3'] = -1


    # Working on df_train
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    # Working on df_test
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    row_ids = df_cell_test.index

    ##
    for index, row in cv_grid.iterrows():
        print(str(index) + "-delta_time:" + str(row['delta_time']) + "-k:" + str(row['k']))
        fmat = df_cell_train
        fmat['win'] = (fmat.time.values // row['delta_time'])
        wins = np.sort(np.unique(fmat.win.values))
        for x in range(wins.shape[0]):
            fmatt = fmat.loc[fmat.win == wins[x]]
            print(str(fmatt))
            print("**********************************")

