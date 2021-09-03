# add need lib
import numpy as np
import pandas as pd
import math
import missingno as msno
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set2')

import os

# download csv file
df = pd.read_csv('sensors.csv')

# let's look at the data
df
