import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

f_data = pd.read_csv("airplane-clean.csv", index_col = 0)
f_data = f_data.dropna()
print f_data.shape
f_data = f_data[f_data.apply(lambda x: len(x['MODEL']) <= 25 and x['YEAR'] != 'None', axis=1)]
print f_data.shape
f_data.dropna()
print f_data.shape
f_data.to_csv('data/planes_cleaned.csv')

plane_data = pd.read_csv("data/planes_cleaned.csv", index_col = 0)
plane_data.head()
print plane_data.shape

flight_data = pd.read_csv("data/flights-limited.csv", index_col = 0)
flight_data.head()
print flight_data.shape

model_data = flight_data.merge(plane_data, on = 'TAIL_NUMBER')
print model_data.shape
sample_size = 29500
early = model_data[model_data['ARRIVAL_DELAY'] == 0]
print early.shape
on_time = model_data[model_data['ARRIVAL_DELAY'] == 1]
print on_time.shape
delayed = model_data[model_data['ARRIVAL_DELAY'] == 2]
print delayed.shape

early = early.sample(sample_size)
on_time = on_time.sample(sample_size)
delayed = delayed.sample(sample_size)
print early.shape
print on_time.shape
print delayed.shape

model_data = pd.concat([early, on_time, delayed])
print model_data.shape
model_data.to_csv('data/model_data.csv')
features = model_data
target = features[['ARRIVAL_DELAY']]
features.drop(labels = ['ARRIVAL_DELAY'], axis = 1, inplace = True)
features.head()
train_x, test_x, train_y, test_y = train_test_split(features, target, train_size=0.8)
print train_x.shape
print train_y.shape
print test_x.shape
print test_y.shape

train = pd.concat([train_x, train_y], axis = 1)
test = pd.concat([test_x, test_y], axis = 1)
train.to_csv('data/model-train.csv')
test.to_csv('data/model-test.csv')