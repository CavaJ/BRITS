import xgboost as xgb
import numpy as np
np.random.seed(1)
import pandas
import utils
from datetime import datetime

dateTimeObj = datetime.now()
print(dateTimeObj)


model_name = 'gru_d' # 'gru_d' # 'm_rnn' # 'brits'

loaded = np.load('./result/{}_data_nosampled.npy'.format(model_name))
label = np.load('./result/{}_label_nosampled.npy'.format(model_name))

print("Loaded data shape: " + str(loaded.shape))  # (3997, 48, 35)
# TODO somehow also get and places record ids to the first dimension, and save the file in text format


new_loaded = np.nan_to_num(loaded)
print(new_loaded.shape)
#save the data in .txt format
utils.saveData(new_loaded, label, 554, 3443)
exit(0)






# reshape data for random forest
impute = loaded.reshape(-1, 48 * 35)

data = np.nan_to_num(impute)

# n_train = 1108  # number of training instances, there are 1108 instances in set-a

#print(impute.shape)
print("Data shape: " + str(data.shape))
print("Label shape: " + str(label.shape))

index = 0
train = []
train_labels = []
test = []
test_labels = []
pos_count = 0
neg_count = 0
for mtse in data:
    #print(mtse.shape, label.reshape(-1,)[index])

    if(label.reshape(-1,)[index] == 1 and pos_count < 554):
        train.append(mtse)
        train_labels.append(1)
        pos_count += 1
    elif(label.reshape(-1,)[index] == 0 and neg_count < 3443): #554):
        train.append(mtse)
        train_labels.append(0)
        neg_count += 1
    else:
        test.append(mtse)
        test_labels.append(label.reshape(-1,)[index])
    index += 1


train = np.asarray(train)
train_labels = np.asarray(train_labels)
test = np.asarray(test)
test_labels = np.asarray(test_labels)

print("Train shape: " + str(train.shape))
print("Train label shape: " + str(train_labels.shape))
print("Test shape: " + str(test.shape))
print("Test label shape: " + str(test_labels.shape))
#exit(0)


utils.countPosCountNeg(train_labels)
utils.countPosCountNeg(test_labels)


# exit(0)


#test = []
#mest = []
#mest.append(1)
#mest.append(2)
#for i in range(5):
#    test.append(mest)

#print(test)
#for mest in test:
#    print("a -> " + str(mest[0]), "b -> " + str(mest[1]))

#print(np.mean(test))

#exit(0)


#print("impute is : ")
#print(impute)

#print("data is : ")
#row_labels = range(3997)
#column_labels = range(1680)
#print pandas.DataFrame(data, columns=column_labels, index=row_labels)

#'''
#exit(0)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, average_precision_score

auc = []
ap = []

for i in range(10):
    # set seed for random forest
    # n_estimators - the number of trees in the forest
    model = RandomForestClassifier(n_estimators=100, random_state=np.random.seed(i)).fit(train, train_labels)
    pred = model.predict_proba(test)

    rfc_pred = np.argmax(pred, axis=1)

    #print(label[n_train:].reshape(-1,).tolist())
    #print(rfc_pred.tolist())

    labels_1d = test_labels.reshape(-1,)

    cur_auc = roc_auc_score(labels_1d, pred[:, 1].reshape(-1, ))
    cur_ap = average_precision_score(labels_1d, pred[:, 1].reshape(-1, ))

    conf_matrix = confusion_matrix(labels_1d, rfc_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    print("=== Confusion Matrix ===")
    print(conf_matrix)
    print("tp = " + str(tp), "fn = " + str(fn), "fp = " + str(fp), "tn = " + str(tn))
    #print('\n')
    #print("=== Classification Report ===")
    #print(classification_report(labels_1d, rfc_pred))
    #print('\n')

    print("Run " + str(i+1) + ": ", "AUROC: " + str(cur_auc), ", AUPRC: " + str(cur_ap))
    auc.append(cur_auc)
    ap.append(cur_ap)

print("Mean AUROC: ", np.mean(auc))
print("Mean AUPRC: ", np.mean(ap))
print("Max AUROC: ", np.amax(auc))
print("Max AUPRC: ", np.amax(ap))
#'''
#exit(0)

dateTimeObj = datetime.now()
print(dateTimeObj)


