import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# read in files & filter out blank lines
print 'reading in data'
full_dataset = pd.read_csv("~/Documents/Junior/CS221/flight-delays/data/flights.csv")
full_dataset = full_dataset[np.isfinite(full_dataset['ARRIVAL_DELAY'])]
print 'done reading data'

# sample a portion of the data set b/c the dataset contains 5M+ rows
sample_size = 100000
sampled_data = full_dataset.sample(sample_size)

baseline_features = sampled_data[['AIRLINE','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE']]
baseline_yx = sampled_data[['ARRIVAL_DELAY', 'AIRLINE','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE']]

# encode arrival delays into discretized buckets, printing progress every 1% of the way
print 'starting update'
for row in range(baseline_yx.shape[0]):
    if (row + 1) % (sample_size/100) == 0:
        print 'updated ' + str(row + 1)
    delay = baseline_yx.iloc[row, 0]
    if delay < 0:
        baseline_yx.iloc[row, 0] = 0
    elif delay <= 20:
        baseline_yx.iloc[row, 0] = 1
    elif delay <= 40:
        baseline_yx.iloc[row, 0] = 2
    elif delay <= 60:
        baseline_yx.iloc[row, 0] = 3
    elif delay <= 80:
        baseline_yx.iloc[row, 0] = 4
    elif delay <= 100:
        baseline_yx.iloc[row, 0] = 5
    elif delay <= 120:
        baseline_yx.iloc[row, 0] = 6
    else:
        baseline_yx.iloc[row, 0] = 7

encoded_target = baseline_yx[['ARRIVAL_DELAY']]
encoded_baseline_features = pd.concat([baseline_features.drop('AIRLINE', axis=1), pd.get_dummies(baseline_features['AIRLINE'])], axis=1)
encoded_baseline_features = pd.concat([encoded_baseline_features, pd.get_dummies(baseline_features['ORIGIN_AIRPORT'])], axis=1)
encoded_baseline_features = pd.concat([encoded_baseline_features, pd.get_dummies(baseline_features['DESTINATION_AIRPORT'])], axis=1)
encoded_baseline_features = encoded_baseline_features.drop('ORIGIN_AIRPORT', axis = 1)
encoded_baseline_features= encoded_baseline_features.drop('DESTINATION_AIRPORT', axis = 1)

# oracle model
oracle_yx = sampled_data[['ARRIVAL_DELAY','DAY', 'MONTH', 'AIRLINE','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']]
encoded_oracle_features = pd.concat([encoded_baseline_features, sampled_data['DEPARTURE_DELAY']], axis=1)
train_xo, test_xo, train_yo, test_yo = train_test_split(encoded_oracle_features, encoded_target, train_size=0.8)
lr_oracle = LogisticRegression()
lr_oracle.fit(train_xo, train_yo)
lr_train_acc = metrics.accuracy_score(train_yo, lr_oracle.predict(train_xo))
lr_test_acc = metrics.accuracy_score(test_yo, lr_oracle.predict(test_xo))
print ('oracle train acc: ' + str(lr_train_acc))
print ('oracle test acc: ' + str(lr_test_acc))

pred_y_testo = lr_oracle.predict(test_xo)
pred_y_testo = [1 if pred_y_testo[i] > 1 else 0 for i in range(len(pred_y_testo))]
test_y_listo = test_yo.as_matrix().tolist()
test_y_binarizedo = [1 if test_y_listo[i][0] > 1 else 0 for i in range(len(test_y_listo))]
print ('oracle precision: ' + str(metrics.precision_score(test_y_binarizedo, pred_y_testo)))

# baseline model
train_xb = train_xo.drop('DEPARTURE_DELAY', axis=1)
train_yb = train_yo
test_xb = test_xo.drop('DEPARTURE_DELAY', axis = 1)
test_yb = test_yo

lr = LogisticRegression()
lr.fit(train_xb, train_yb)
lr_train_acc = metrics.accuracy_score(train_yb, lr.predict(train_xb))
lr_test_acc = metrics.accuracy_score(test_yb, lr.predict(test_xb))
pred_y_test = list(lr.predict(test_xb))
pred_y_test = [1 if pred_y_test[i] > 1 else 0 for i in range(len(pred_y_test))]
test_y_list = test_yb.as_matrix().tolist()
test_y_binarized = [1 if test_y_list[i][0] > 1 else 0 for i in range(len(test_y_list))]
print ('baseline train acc: ' + str(lr_train_acc))
print ('baseline test acc: ' + str(lr_test_acc))
print ('baseline precision: ' + str(metrics.precision_score(test_y_binarized, pred_y_test, average = 'weighted')))

# confusion matrix
y_true = test_yb
y_pred = lr.predict(test_xb)
confusion = confusion_matrix(y_true, y_pred)
print confusion
