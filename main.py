# need lib
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# download features
data_x = pd.read_csv('sensors.csv', parse_dates=['timestamp'])

# download target
data_y = pd.read_csv('coke_target.csv', parse_dates=['timestamp'])

# download base target
base_target = pd.read_csv('baseline_coke.csv', parse_dates=['timestamp'])

# merge dataframes
data = data_x.join(data_y.set_index('timestamp'), on='timestamp')

# replace the missing values with the median values
for col in data.columns[1:]:
  data[col].fillna(data[col].median(), inplace=True)

# data for regressor
X = data.drop(['timestamp', 'target'], axis=1)
y = data.target
# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=17)

# creating a LinearRegression classifier, training, evaluating accuracy
lr = linear_model.LinearRegression(normalize=True)
lr = lr.fit(X_train, y_train)
lr_score = mean_squared_error(y_test, lr.predict(X_test))
print('mean_squared_error: ', lr_score)

# add column predicted
pred = lr.predict(data.drop(['timestamp', 'target'], axis=1))
data['predict'] = pred

# graf with target value and predict
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['target'],
                    mode='lines',
                    name='target'))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['predict'],
                    mode='lines',
                    name='predict'))

pio.show(fig)

# error compared to the basic solution
mean_squared_error_base = mean_squared_error(base_target.target,
                   lr.predict(data[data['timestamp']>='2018-01-01 00:00:00']
                              .drop(['timestamp', 'target', 'predict'], axis=1)))

print('mean_squared_error_base: ', mean_squared_error_base)