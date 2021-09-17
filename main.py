# need lib
import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score, TimeSeriesSplit
from itertools import product
from tqdm import tqdm_notebook
from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import warnings


# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show()

# download features
data_x = pd.read_csv('sensors.csv', parse_dates=['timestamp'])

# download target
data_y = pd.read_csv('coke_target.csv', parse_dates=['timestamp'])

# download target for analysis with SARIMAX
target = pd.read_csv('coke_target.csv', index_col=['timestamp'], parse_dates=['timestamp'])
target = target.resample('1D').median() # reindexing in the case of a large sample
print(target.head(3))

# let's look at the graph of the target variable
plt.figure(figsize=(18, 6))
plt.plot(target.target)
plt.title('Target watched (hourly data)')
plt.grid(True)
plt.show()

# Let's build a time series graph, its ACF and PACF, calculate the Dickey-Fuller test
tsplot(target.target, lags=100)

# seasonal diff
target_diff = target.target - target.target.shift(7)
tsplot(target_diff[7:], lags=100)

# without seasonal diff
target_diff = target_diff - target_diff.shift(1)
tsplot(target_diff[7+1:], lags=100)

# initializing groups of parameters for substitution
ps = range(1, 5)
d=2
qs = range(1, 5)
Ps = range(0, 2)
D=1 # D is equal to 1, because we take into account the seasonal difference
Qs = range(0, 2)
s = 7 # len season 7

# create a list with all possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


def optimizeSARIMA(parameters_list, d, D, s):
    """Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(target.target, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table

# calculations for all combinations of parameters
warnings.filterwarnings("ignore")
result_table = optimizeSARIMA(parameters_list, d, D, s)

# set the parameters with the smallest AIC
p, q, P, Q = result_table.parameters[0]

best_model=sm.tsa.statespace.SARIMAX(target.target, order=(p, d, q),
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())

# build graphs for the model with the best parameters
tsplot(best_model.resid[7+1:], lags=100)


def plotSARIMA(series, model, n_steps):
    """Plots model vs predicted values

        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    """

    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:s + d] = np.NaN

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.sarima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s + d:], data['sarima_model'][s + d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    plt.show()

# overlay the graphs of the target variable and the predictions on top of each other
plotSARIMA(target, best_model, 50)

# download base target
base_target = pd.read_csv('baseline_coke.csv', parse_dates=['timestamp'])

# merge dataframes
data = data_x.join(data_y.set_index('timestamp'), on='timestamp')

# replace the missing values with the rolling window calculations
data.target[data['timestamp'] >= '2018-01-01 00:00:00'] = data.target[data['timestamp'] <= '2018-01-01 00:00:00'][:2872]
for col in data.columns[1:]:
    data[col].fillna(data[col].median(), inplace=True)

# data for regressor
X = data.drop(['timestamp', 'target'], axis=1)
y = data.target
# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=17)

# creating a LinearRegression classifier, training, evaluating accuracy
#lr = linear_model.LinearRegression(normalize=True)
lr = LassoCV(cv=5, random_state=17)
lr = lr.fit(X_train, y_train)
lr_score = sklearn.metrics.mean_squared_error(y_test, lr.predict(X_test))
print('mean_squared_error: ', lr_score)

# add column predicted
pred = lr.predict(data.drop(['timestamp', 'target'], axis=1))
data['predict'] = pred

mean_squared_error_base = sklearn.metrics.mean_squared_error(base_target.target,
                                                             lr.predict(data[data['timestamp'] >= '2018-01-01 00:00:00']
                                                        .drop(['timestamp', 'target', 'predict'], axis=1)))

print('mean_squared_error_base: ', mean_squared_error_base)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = lr.predict(X_test)#data[data['timestamp'] >= '2018-01-01 00:00:00']
                                #                        .drop(['timestamp', 'target', 'predict'], axis=1))#cross_val_predict(lr, X, y, cv=5)

fig, ax = plt.subplots()
ax.scatter(y_test, predicted, edgecolors=(0, 1, 0))
ax.plot([predicted.min(), predicted.max()], [predicted.min(), predicted.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()