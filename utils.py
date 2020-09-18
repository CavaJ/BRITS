import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
np.random.seed(1)

from torch.autograd import Variable

import pandas as pd

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x




def saveData(data_, labels, set_a_pos_count, set_a_neg_count):
    idx = 0
    tr = []
    tr_labels = []
    tst = []
    tst_labels = []
    p_count = 0
    n_count = 0
    for mtse_ in data_:
        # print(mtse.shape, label.reshape(-1,)[index])

        if (labels.reshape(-1, )[idx] == 1 and p_count < set_a_pos_count):
            tr.append(mtse_)
            tr_labels.append(1)
            p_count += 1
        elif (labels.reshape(-1, )[idx] == 0 and n_count < set_a_neg_count):
            tr.append(mtse_)
            tr_labels.append(0)
            n_count += 1
        else:
            tst.append(mtse_)
            tst_labels.append(labels.reshape(-1, )[idx])
        idx += 1

    attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose',
                  'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
                  'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
                  'Creatinine', 'ALP']

    # TRAINING DATA
    for j in range(len(tr)):
        res = 'tsMinutes'
        for i in range(len(attributes)):
            res += "," + str(attributes[i])
        res += "\n"

        tse = tr[j]
        initts = 60
        for ts in tse:
            res += str(initts)
            for vl in ts:
                res += "," + str(vl)
            res += "\n"
            initts += 60

        fs = open("./set_a/{}.txt".format(j+1), "w")
        fs.write(res)
        print(str(j+1) + ".txt is written")
        fs.close()

    res = ""
    for k in range(len(tr_labels)):
        res += str(k+1) + "," + str(tr_labels[k]) + "\n"
    fs = open("./set_a/Outcomes_set_a.txt", "w")
    fs.write(res)
    print("Outcomes file for training data is written")
    fs.close()


    # TEST DATA
    for j in range(len(tst)):
        res = 'tsMinutes'
        for i in range(len(attributes)):
            res += "," + str(attributes[i])
        res += "\n"

        tse = tst[j]
        initts = 60
        for ts in tse:
            res += str(initts)
            for vl in ts:
                res += "," + str(vl)
            res += "\n"
            initts += 60

        fs = open("./set_b/{}.txt".format(j+1 + (set_a_pos_count + set_a_neg_count)), "w")
        fs.write(res)
        print(str(j+1 + (set_a_pos_count + set_a_neg_count)) + ".txt is written")
        fs.close()

    res = ""
    for k in range(len(tst_labels)):
        res += str(k+1 + (set_a_pos_count + set_a_neg_count)) + "," + str(tst_labels[k]) + "\n"
    fs = open("./set_b/Outcomes_set_b.txt", "w")
    fs.write(res)
    print("Outcomes file for test data is written")
    fs.close()


    tr = np.asarray(tr)
    tr_labels = np.asarray(tr_labels)
    tst = np.asarray(tst)
    tst_labels = np.asarray(tst_labels)

    print("Train shape: " + str(tr.shape))
    print("Train label shape: " + str(tr_labels.shape))
    print("Test shape: " + str(tst.shape))
    print("Test label shape: " + str(tst_labels.shape))



def countPosCountNeg(labels):
    t_labels = labels.reshape(-1, )
    count_pos = 0
    count_neg = 0
    for label in t_labels:
        if label == 1:
            count_pos = count_pos + 1
        else:
            count_neg = count_neg + 1
    print("count_pos: " + str(count_pos), "count_neg: " + str(count_neg))