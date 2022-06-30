import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn import model_selection


def pre_process(data):
    """
    Redefine the variable type, turn binary-valued categories into binary
    value 0 and 1; represent multi-value categories by numbers 0, 1, 2, 3...
    :param data: The original dataset
    :return: The re-casted dataset
    """
    map_sch = {'GP': 1, 'MS': 0}
    map_sex = {'M': 1, 'F': 0}
    map_add = {'U':1, 'R':0}
    map_fam = {'LE3':1, 'GT3':0}
    map_pat = {'T':1, 'A':0}
    map_job = {'teacher':4, 'health':3, 'services':2, 'at_home':1, 'other':0}
    map_rea = {'home':3,'reputation':2, 'course':1, 'other':0}
    map_gua = {'mother':2, 'father':1, 'other':0}
    map_yn = {'yes':1, 'no':0}

    data['school'] = data['school'].map(map_sch)
    data['sex'] = data['sex'].map(map_sex)
    data['address'] = data['address'].map(map_add)
    data['famsize'] = data['famsize'].map(map_fam)
    data['Pstatus'] = data['Pstatus'].map(map_pat)
    data['Mjob'] = data['Mjob'].map(map_job)
    data['Fjob'] = data['Fjob'].map(map_job)
    data['reason'] = data['reason'].map(map_rea)
    data['guardian'] = data['guardian'].map(map_gua)
    data['schoolsup'] = data['schoolsup'].map(map_yn)
    data['famsup'] = data['famsup'].map(map_yn)
    data['paid'] = data['paid'].map(map_yn)
    data['activities'] = data['activities'].map(map_yn)
    data['nursery'] = data['nursery'].map(map_yn)
    data['higher'] = data['higher'].map(map_yn)
    data['internet'] = data['internet'].map(map_yn)
    data['romantic'] = data['romantic'].map(map_yn)
    return data


def data_normalization(data):
    """
    The pre-process method: dataset regularization.
    :param data: The original dateset
    :return: The dataset after preprocess
    """
    # nor = MinMaxScaler()
    # nor = StandardScaler()
    nor = Normalizer()
    data  = nor.fit_transform(data)
    return data



def get_data_and_label(tr_set, te_set):
    """
    Divide the training data into training dataset and validation dataset
    Get the features and labels from the datasets.
    :param tr_set: The training dataset
    :param te_set: The test dataset
    :return: The features and labels of training data, validation data and
             test data.
    """
    # tr_set = tr_set.values.tolist()
    # te_set = te_set.values.tolist()
    tr_set = np.array(tr_set)
    te_set = np.array(te_set)

    # np.random.shuffle(tr_set)
    # np.random.shuffle(te_set)

    tr_set, val_set = model_selection.train_test_split(tr_set, test_size=0.3)

    tr_data = tr_set[:, 0:-3]
    tr_labs = tr_set[:, -3:]

    val_data = val_set[:, 0:-3]
    val_labs = val_set[:, -3:]

    te_data = te_set[:, 0:-3]
    te_labs = te_set[:, -3:]

    return tr_data, te_data, val_data, tr_labs, val_labs, te_labs


def problem_num(tr_data, te_data, val_data, tr_labs, val_labs, te_labs, num):
    """
    Get different final input based on different problems.
    :param tr_data: The features of training data
    :param te_data: The features of test data
    :param val_data: The features of validation data
    :param tr_labs: The labels of training set
    :param val_labs: The labels of validation set
    :param te_labs: The labels of test set
    :param num: The index of problem
    :return: New features and labels of different problems
    """
    if num == 1:
        tr_x = tr_data
        tr_y = tr_labs[:, 0]
        val_x = val_data
        val_y = val_labs[:, 0]
        te_x = te_data
        te_y = te_labs[:, 0]
    elif num == 2:
        tr_x = tr_data
        tr_y = tr_labs[:, 2]
        val_x = val_data
        val_y = val_labs[:, 2]
        te_x = te_data
        te_y = te_labs[:, 2]
    elif num == 3:
        tr_x = np.c_[tr_data,tr_labs[:,0:2]]
        tr_y = tr_labs[:, 2]
        val_x = np.c_[val_data, val_labs[:,0:2]]
        val_y = val_labs[:, 2]
        te_x = np.c_[te_data,te_labs[:,0:2]]
        te_y = te_labs[:, 2]
    return tr_x, tr_y, val_x, val_y, te_x, te_y
