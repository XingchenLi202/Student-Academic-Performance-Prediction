import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def perf_measures(pre_y, targ_y):
    """
    RMSE: Root mean squared error
    MAE: Mean absolute error
    R-square: Coefficient of determination
    :param pre_y: The predicted labels
    :param targ_y: The target labels
    :return: Three algorithmic measures
    """
    rmse = mean_squared_error(targ_y, pre_y, squared=False)
    mae = mean_absolute_error(targ_y, pre_y)
    r2 = r2_score(targ_y, pre_y)
    # rmse = np.sqrt(((pre_y - targ_y) ** 2).mean())
    # mae = np.abs(targ_y-pre_y).mean()
    # SSE = sum((pre_y - targ_y) ** 2)
    # SST = sum((targ_y.mean()-targ_y)**2)
    # r2 = 1 - SSE/SST
    return rmse, mae, r2


def cross_val_comp(models, tr_x, tr_y, num):
    """
    Compare cross validation performance of different models
    :param models: The models we used
    :param tr_x: The feature of dataset
    :param tr_y: The label of dataset
    :param num: Specify the number of folds
    :return: Three validation indicators of different models
             and their accuracies
    """
    frame = pd.DataFrame()
    rmses = []
    maes = []
    r2s = []
    accuracies = []

    for model in models:
        # calculate RMSE
        mse = cross_val_score(model, tr_x, tr_y,
                              scoring='neg_mean_squared_error', cv=num)
        mse = -np.round(mse, 5)
        rmse = np.sqrt(mse)
        rmses.append(rmse)
        rmse_average = np.round(rmse.mean(), 5)

        # calculate MAE
        mae = cross_val_score(model, tr_x, tr_y,
                              scoring='neg_mean_absolute_error', cv=num)
        mae = -np.round(mae, 5)
        maes.append(mae)
        mae_average = mae.mean()
        mae_average = np.round(mae_average, 5)

        # calculate R-square
        r2 = cross_val_score(model, tr_x, tr_y, scoring='r2', cv=num)
        r2 = np.round(r2, 5)
        r2s.append(r2)
        r2_average = r2.mean()
        r2_average = np.round(r2_average, 5)

        # calculate accuracy
        accuracy = 100 * (1- len(tr_x)*mae / sum(tr_y))
        accuracy = np.round(accuracy, 5)
        accuracies.append(accuracy)
        accuracy_average = accuracy.mean()
        accuracy_average = np.round(accuracy_average, 5)

        frame[str(model)] = [rmse_average, mae_average,
                             r2_average, accuracy_average]
        # frame[str(model)] = [max(rmses),max(maes),max(r2s),max(accuracies)]
        frame.index = ['Root Mean Squared Error', 'Mean Absolute Error',
                       'R^2', 'Accuracy']
    return frame, rmses, maes, r2s, accuracies



def get_best_fold(models, tr_x, tr_y, num):
    """
    get the best number of fold.
    :param models: the list of models
    :param tr_x: The feature of dataset
    :param tr_y: The label of dataset
    :param num: Specify the number of folds
    :return: the r2 result of every fold
    """
    frame, rmses, maes, r2s, accuracies = cross_val_comp(models, tr_x, tr_y, num)
    r2_fold = pd.DataFrame(r2s, index=frame.columns,
                           # columns=['1 fold', '2 fold', '3 fold',
                           #          '4 fold', '5 fold']
                           )
    r2_fold['Ave_R2'] = r2_fold.mean(axis=1)
    np.round(r2_fold, 5)

    trans = pd.DataFrame(r2_fold.values.T, columns=frame.columns)
    trans.plot()
    # r2_fold.plot()
    plt.xlabel("The number of CV")
    plt.ylabel("R-square")
    plt.grid()
    plt.show()
    return r2_fold