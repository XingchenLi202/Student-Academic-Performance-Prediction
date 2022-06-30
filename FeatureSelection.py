import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

def pca(tr_x, te_x):
    """
    Reduce the dimension of feature space to get new features.
    :param tr_x: The features of data
    :param te_x: The features of the dataset that we want to process
    :return: New features after reducing dimension
    """
    pca = PCA(n_components=12)
    pca.fit(tr_x)
    # pca.fit_transform(te_x)
    return tr_x, te_x


def remove_features(X):
    """
    Remove 4 categorical non-binary features.
    :param X: The features of dataset
    :return: The feature set after removing
    """
    X_new = np.hstack((X[:,:8], X[:,12:]))
    return X_new


def feature_sel(X,y,k):
    """
    According to the dataset and the number of feature, get the final k best
    features.
    :param X: The features of data
    :param y: The labels of data
    :param k: The number of features that we want to get finally
    :return: The indices of k best features
    """
    kb = SelectKBest(f_regression, k=k)
    kb.fit_transform(X,y)
    scores = kb.scores_
    idx = np.argsort(scores)[::-1]
    X = pd.DataFrame(X)
    k_best = X.columns
    k_best = k_best.values[idx[0:k]]
    k_best = list(k_best)
    return k_best


def get_best_feature_number(models, tr_x, tr_y):
    """
    Based on cross-validation, plot the trend of R-square with the
    increasing number of top k best feature, and finally get the best
    number of feature.
    :param models: The list of models
    :param tr_x: The original feature
    :param tr_y: The target label
    """
    k_range = range(1,30)
    for m in models:
        scor_of_model = []
        for k in k_range:
            k_best = feature_sel(tr_x,tr_y,k)
            new_x = get_feature(tr_x, k_best)
            scores = cross_val_score(m, new_x, tr_y, cv=5, scoring='r2')
            scor_of_model.append(scores.mean())
        # print(scor_of_model)
        plt.plot(k_range, scor_of_model)
        plt.xlabel(str(m))
        plt.ylabel('Cross-Validated R2')
        plt.grid()
        plt.show()


def get_feature(X, k_best):
    """
    According to the indices of k best features get the new feature matrix.
    :param X: The original feature matrix
    :param k_best: The indices of k best features
    :return: The new feature matrix
    """
    X_new = []
    for i in k_best:
        X_new.append(X[:,i])
    X_new = np.array(X_new)
    X_new = np.transpose(X_new)
    return X_new


