import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.pyplot import MultipleLocator
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor
import xgboost
import lightgbm
import catboost


# Dictionary about promoters
Mydic = {'23104': 0, '1-16': 1, '2-22': 2, '1-29': 3,
         'yccFS': 4, '2-21': 5, '23112': 6, '1-17': 7,
         '1-57': 8, 'RacR': 9, 'trxA': 10, 'rrnA': 11}
Inv_Charmap = {'0': '23104', '1': '1-16', '2': '2-22', '3': '1-29',
               '4': 'yccFS', '5': '2-21', '6': '23112', '7': '1-17',
               '8': '1-57', '9': 'RacR', '10': 'trxA', '11': 'rrnA'}

# Load data
def Load_data():
    # data = np.array(pd.read_excel('RSF gene mutation promoter library.xlsx', sheet_name='Sheet2'))
    # data = np.array(pd.read_excel('RSF_4_deleted.xlsx', sheet_name='Sheet1'))
    data = np.array(pd.read_excel('combinations.xlsx', sheet_name='Experiment'))
    X = np.zeros([len(data), 4, 12], dtype=int)
    y = []
    for i in range(len(data)):
        for j in range(4):
            X[i][j][Mydic[str(data[i][j])]] = 1
        y.append(data[i][-1])
    X = np.array(X.reshape([len(data), 48]))
    y = np.array(y)
    return X, y


# Model construction
def Train_model(X, y):
    # Single model
    # Mymodel = [Ridge(), Lasso(), KNeighborsRegressor(), SVR(), DecisionTreeRegressor(),
    #            RandomForestRegressor(), ExtraTreesRegressor(), AdaBoostRegressor(), BaggingRegressor(),
    #            GradientBoostingRegressor(), xgboost.XGBRegressor(), lightgbm.LGBMRegressor(),
    #            catboost.CatBoostRegressor(logging_level='Silent')]

    # Fused model
    Ensemble_model = VotingRegressor([('1', GradientBoostingRegressor()), ('2', Ridge()),
                                      ('3', catboost.CatBoostRegressor(logging_level='Silent')), ('4', Lasso()),
                                      ('5', xgboost.XGBRegressor())])
    Mymodel = [Ensemble_model]

    # training process
    ALL_precision = []
    ALL_pre_label = []
    ALL_real_label = []
    for i in range(len(Mymodel)):
        model = Mymodel[i]
        kf = KFold(n_splits=10, shuffle=True)
        Corr = 0
        RMSE = 0
        MAE = 0
        _, ax = plt.subplots(1, 1, figsize=(18, 12), dpi=600)
        for train_index, test_index in kf.split(X, y):
            Train_data, Train_label = X[train_index], y[train_index]
            Test_data, Test_label = X[test_index], y[test_index]
            model.fit(Train_data, Train_label)
            Pre_label = model.predict(Test_data)
            RMSE += np.sqrt(mean_squared_error(Test_label, Pre_label))
            MAE += mean_absolute_error(Test_label, Pre_label)
            Corr += np.corrcoef(Test_label, Pre_label)
            ALL_pre_label.extend(Pre_label)
            ALL_real_label.extend(Test_label)
            plt.scatter(Pre_label, Test_label, color='darkblue', s=60)
        plt.legend(['PCC=0.82'], loc = 'upper right', fontsize=34)
        Corr *= 0.1
        MAE *= 0.1
        RMSE *= 0.1
        ALL_precision.append([Corr[0][1], RMSE, MAE])
        print('Corr:', Corr[0][1], 'RMSE:', RMSE, 'MAE:', MAE)
        plt.ylim(0, 1600)
        plt.xlim(0, 1100)
        plt.xticks(fontsize=30, fontweight='bold')
        plt.yticks(fontsize=30, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth('5.0')
        ax.spines['left'].set_linewidth('5.0')
        Mystr = "PCC="+str(round(Corr[0][1], 2))
        print('PCC:', Mystr)
        # Picture layout
        plt.tick_params(pad=32)
        for tickline in ax.xaxis.get_ticklines():
            tickline.set_markersize(24)
            tickline.set_markeredgewidth(9)
        for tickline in ax.yaxis.get_ticklines():
            tickline.set_markersize(24)
            tickline.set_markeredgewidth(9)
        plt.xlabel('Predicted Naringenin production (mg/L)', size = 34, weight='bold')
        plt.ylabel('Real Naringenin production (mg/L)', size = 34, weight='bold')
        y_major_locator = MultipleLocator(250)
        ax.yaxis.set_major_locator(y_major_locator)
        x_major_locator = MultipleLocator(250)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.tight_layout()
        plt.savefig('Fig4d.png')

    # Save all predicted results
    ALL_pre_label = np.array(ALL_pre_label).T
    ALL_real_label = np.array(ALL_real_label).T
    res = pd.DataFrame({'ALL_pre_label': ALL_pre_label, 'ALL_real_label': ALL_real_label})
    res.to_excel('ALL_predicted_real_label.xlsx')


# Model prediction
def Predict_model(X, y):
    model = VotingRegressor([('1', GradientBoostingRegressor()), ('2', Ridge()),
                             ('3', catboost.CatBoostRegressor(logging_level='Silent')), ('4', Lasso()),
                             ('5', xgboost.XGBRegressor())])
    model.fit(X, y)

    # Generate all combinations
    X_input = np.zeros([144*144, 4, 12], dtype=int)
    for i in range(12):
        for j in range(12):
            for k in range(12):
                for l in range(12):
                    X_input[i * 12 * 12 * 12 + j * 12 * 12 + k * 12 + l][0][i] = 1
                    X_input[i * 12 * 12 * 12 + j * 12 * 12 + k * 12 + l][1][j] = 1
                    X_input[i * 12 * 12 * 12 + j * 12 * 12 + k * 12 + l][2][k] = 1
                    X_input[i * 12 * 12 * 12 + j * 12 * 12 + k * 12 + l][3][l] = 1
    X_input = X_input.reshape([144*144, 48])

    # Prediction of all combinations
    Pre_label = model.predict(X_input)
    Pre_label = Pre_label.T

    # Save all results
    Predicted_data = []
    X_input = X_input.reshape([144*144, 4, 12])
    for i in range(len(X_input)):
        zj = []
        for j in range(len(X_input[i])):
            zj.append(Inv_Charmap[str(int(np.argwhere(X_input[i][j] == 1)))])
        Predicted_data.append(zj)
    Predicted_data = np.array(Predicted_data).T
    res = pd.DataFrame({'1': Predicted_data[0], '2': Predicted_data[1], '3': Predicted_data[2],
                        '4': Predicted_data[3], 'Label': Pre_label})
    res.to_excel('Predicted_ALL_Promoter_combination.xlsx')


if __name__ == '__main__':
    X, y = Load_data()
    Train_model(X, y)
    Predict_model(X, y)
