# add need lib
import numpy as np
import pandas as pd
import math
import missingno as msno
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set2')

import os

# download csv file
df_sensors = pd.read_csv('sensors.csv', parse_dates=['timestamp'])
df_target = pd.read_csv('coke_target.csv', parse_dates=['timestamp'])

# merge dataframes
df = df_sensors.join(df_target.set_index('timestamp'), on='timestamp')

# let's look at the data
print(df)

# let's look at the presence of missing values
msno.bar(df)
msno.matrix(df)

# we will output statistics for each column
print(df.dtypes)
print(df.describe(include=[np.number])) # number columns
print(df.describe(include=[np.object])) # not number

# Nan values -> median
for col in df.columns[1:]:
  df[col].fillna(df[col].median(), inplace=True)

# pairwise correlation
correlation = df.corr()
plt.figure(figsize=(35,35))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

plt.title('Correlation between different features')
plt.show()

# all graf
fig = go.Figure()
for col in df.columns[1:]:
  fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col],
                    mode='lines',
                    name=col))

fig.show()