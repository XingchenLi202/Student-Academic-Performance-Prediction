import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def model_KNN(tr_x, tr_y, te_x,k):
    """
    K-Nearest-Neighbors Algorithm
    :param tr_x: The features of training data
    :param tr_y: The labels of training data
    :param te_x: The features of the dataset that we want to predict
    :param k: The number of neighbors in KNN
    :return: The predicted label after KNN
    """
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(tr_x, tr_y)
    knn_pre_y = knn.predict(te_x)
    return knn_pre_y,knn



def model_LinearRegre(tr_x, tr_y, te_x):
    """
    Linear Regression Algorithm
    :param tr_x: The features of training data
    :param tr_y: The labels of training data
    :param te_x: The features of the dataset that we want to predict
    :return: The predicted label after Linear Regression
    """
    lr = LinearRegression()
    lr.fit(tr_x, tr_y)
    lr_pre_y = lr.predict(te_x)
    return lr_pre_y,lr


def model_trivial_sys(train_y, test_y):
    """
    Trivial System
    :param train_y: The labels of training data
    :param test_y: The target labels of a dataset
    :return: The predicted label of the dataset
    """
    pred_y = np.full(np.shape(test_y), np.mean(train_y))
    return pred_y


def model_SVR(tr_x, tr_y, x):
    """
    Support Vector Regression Algorithm
    :param tr_x: The feature of training data
    :param tr_y: The label of training data
    :param x: The feature of data that we want to predict
    :return: The predicted label of the dataset and the model
    """
    svr = SVR()
    svr.fit(tr_x, tr_y)
    svr_pre_y = svr.predict(x)
    return svr_pre_y, svr


def model_randomforest(tr_x, tr_y, x, rf_p):
    """
    Random Forest Regression Algorithm
    :param tr_x: The feature of training data
    :param tr_y: The label of training data
    :param x: The feature of data that we want to predict
    :return: The predicted label of the dataset and the model
    """
    rfr = RandomForestRegressor(n_estimators=rf_p)
    rfr.fit(tr_x, tr_y)
    rfr_pre_y = rfr.predict(x)
    return rfr_pre_y, rfr

def model_ridge(X, y, val_x, r_p):
    """
    Ridge Regression
    :param X: The feature of training data
    :param y: The label of training data
    :param val_x: The feature of data that we want to predict
    :return: The predicted label of the dataset and the model
    """
    r = RidgeCV(alphas=r_p)
    r.fit(X, y)
    r_pre_y = r.predict(val_x)
    return r_pre_y, r


def get_models(tr_x, tr_y, x, knn_p, rf_p, r_p):
    """
    Get all models that I used and its predicted y,
    usually used in comparing the models.
    :param tr_x: The feature of training data
    :param tr_y: The label of training data
    :param x: The feature of data that we want to predict
    :param k: The number of neighbors in KNN
    :return: Two lists of models and predicted y
    """
    knn_pre_y, knn = model_KNN(tr_x, tr_y, x,knn_p)
    lr_pre_y, lr = model_LinearRegre(tr_x, tr_y, x)
    svr_pre_y, svr = model_SVR(tr_x, tr_y, x)
    rfr_pre_y, rfr = model_randomforest(tr_x,tr_y,x,rf_p)
    r_pre_y, r = model_ridge(tr_x, tr_y, x, r_p)
    models = [knn, lr, svr, rfr, r]
    pre_ys = [knn_pre_y, lr_pre_y, svr_pre_y, rfr_pre_y, r_pre_y]
    return models, pre_ys



def model_best_KNN(X, y, val_x):
    """
    Based on cross-validation, find the best number of neighbors k
    plot the trend of R-square of KNN with increasing number of k.
    :param X: The feature of training data
    :param y: The label of training data
    :param val_x: The feature of data that we want to predict
    :return:
    """
    ran = range(1, 31)
    k_sco = []
    for k in ran:
        _, knn = model_KNN(X,y,val_x,k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='r2')
        k_sco.append(scores.mean())
    plt.plot(ran, k_sco)
    plt.xlabel('Value of K for KNN')#19
    plt.ylabel('Cross-Validated R2')
    plt.grid()
    plt.show()


def model_best_RF(tr_x, tr_y):
    """
    Based on cross-validation, find the best parameter n_estimators of
    RF regression.
    :param tr_x: The feature of training data
    :param tr_y: The label of training data
    :return: m2 is the maximum R-square value, m_idx2 is its index
    """
    n_estim_scores = []
    for n_estim in range(1, 100, 5):
        rf = RandomForestRegressor(n_estimators=n_estim)
        score = cross_val_score(rf, tr_x, tr_y, cv=5, scoring='r2')
        n_estim_scores.append(score.mean())
    m = max(n_estim_scores)
    m_idx = n_estim_scores.index(m)*5+1
    # print(m, m_idx)
    # plt.plot(range(1, 100, 5), n_estim_scores)
    # plt.xlabel('Value of n_estimators for RF')
    # plt.ylabel('Cross-Validated R2')
    # plt.show()

    n_estim_scores2 = []
    for n_estim in range(m_idx-5, m_idx+5, 1):
        rf = RandomForestRegressor(n_estimators=n_estim)
        score = cross_val_score(rf, tr_x, tr_y, cv=5, scoring='r2')
        n_estim_scores2.append(score.mean())
    m2 = max(n_estim_scores2)
    m_idx2 = m_idx-5 + n_estim_scores2.index(m2)
    # print(m2, m_idx2)
    # plt.plot(range(m_idx-5, m_idx+5, 1), n_estim_scores2)
    # plt.xlabel('Value of n_estimators for RF')
    # plt.ylabel('Cross-Validated R2')
    # plt.show()
    return m,m_idx


def model_best_ridge(X, y):
    """
    Based on cross-validation, find the best parameter alphas of
    Ridge regression.
    :param X: The feature of training data
    :param y: The label of training data
    """
    alphas_scores = []
    alphas_range = np.linspace(0.01,1.01)
    for a in alphas_range:
        r = RidgeCV(alphas=a)
        score = cross_val_score(r, X, y, cv=5, scoring='r2')
        alphas_scores.append(score.mean())
    m = max(alphas_scores)
    m_idx = alphas_scores.index(m)*0.02 + 0.01
    print(m, m_idx)
    plt.plot(alphas_range, alphas_scores)
    plt.xlabel('Value of alpha for RidgeCV')
    plt.ylabel('Cross-Validated R2')
    plt.show()
