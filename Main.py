import pandas as pd
import numpy as np
import DataProcess as dp
import FeatureSelection as fs
import Models as ms
import Measurement as mmt


if __name__ == '__main__':
    # get data
    train = pd.read_csv("student_performance_train.csv")
    test = pd.read_csv("student_performance_test.csv")
    train = dp.pre_process(train)
    test = dp.pre_process(test)

    # normalization
    train = dp.data_normalization(train)
    test = dp.data_normalization(test)
    # get data and label
    tr_data, te_data, val_data, tr_lab, val_lab, te_lab \
        = dp.get_data_and_label(train, test)
    # re-merge dataset based on mission number: 1, 2, 3
    tr_x, tr_y, val_x, val_y, te_x, te_y = \
        dp.problem_num(tr_data, te_data, val_data, tr_lab, val_lab, te_lab, 1)

    # remove 4 features
    # tr_x = fs.remove_features(tr_x)
    # val_x = fs.remove_features(val_x)
    # te_x = fs.remove_features(te_x)

    # # get all model
    # models, pre_ys = ms.get_models(tr_x, tr_y, val_x, 1, 100, 0.1) #default

    # get best number of feature
    # fs.get_best_feature_number(models,tr_x,tr_y)

    # select k best features
    k_best = fs.feature_sel(tr_x, tr_y, 15)
    # print(k_best)
    tr_x = fs.get_feature(tr_x, k_best)
    val_x = fs.get_feature(val_x, k_best)
    te_x = fs.get_feature(te_x,k_best)

    # PCA
    # tr_x,te_x = fs.pca(tr_x,te_x)

    # optimal k in KNN: 1.8, 2.5, 3.3
    # ms.model_best_KNN(tr_x, tr_y, val_x)

    # optimal n_estimators in RF
    # m, m_idx = ms.model_best_RF(tr_x, tr_y)

    # optimal alphas in RidgeCV
    # ms.model_best_ridge(tr_x,tr_y)

    # get three measurements of all 5 models
    models, pre_ys = ms.get_models(tr_x, tr_y, te_x, 3, 56, 0.11)
    # print(np.shape(pre_ys))

    # # without cross-validation
    for i in range(np.shape(pre_ys)[0]):
        rmse, mae, r2 = mmt.perf_measures(te_y, pre_ys[i])
        # print("\n")
        # print(rmse)
        # print(mae)
        # print(r2)

    # # get three measurements trivial system
    # ts_val_pre_y = ms.model_trivial_sys(tr_y, val_y)
    # ts_val_rmse, ts_val_mae, ts_val_r2 = mmt.perf_measures(val_y, ts_val_pre_y)
    # print("\n")
    # print(ts_val_rmse,"\n",ts_val_mae,"\n",ts_val_r2)

    # get best number of fold in cross-validation
    mmt.get_best_fold(models,tr_x,tr_y,10)


    # cross validation get better result
    frame, rmses, maes, r2s, accuracies = \
        mmt.cross_val_comp(models, te_x, te_y, 5)
    print(frame)
    # print("\n")
    # print(maes)
    # print("\n")
    # print(rmses)
    # print("\n")
    # print(r2s)
    # print("\n")
    # print(accuracies)





