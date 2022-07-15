#!/usr/bin/env python
# coding: utf-8

# #### Importing Packages

# #### Data Understanding

# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_excel('C:/Users/LENOVO PC/data.xlsx',index_col='Period',parse_dates=True)
data.index.freq = 'MS'
data.head()


# In[24]:


data.tail()


# In[25]:


data.info()


# In[26]:


data.describe()


# #### Data Preparation

# In[27]:


plt.figure(figsize=(12,5))
plt.title('Monthly Passengers Recorded in Ngurah Rai Airport 2006 - 2020')
plt.plot(data,color='green');


# The graph shows that the Passengers data has a seasonal trend until 2018 with a constant ups and downs, but dropped significantly at the start of 2020. Looking back to what happened, this might be because the restriction on outdoor activities due to covid-19 spread, therefore Ngurah Rai has a decreasing number of passengers during that time

# For a better look, we will separate the cyclic and trend pattern of the data

# In[28]:


from statsmodels.tsa.filters.hp_filter import hpfilter
data_cycle, data_trend = hpfilter(data,lamb=1600)


# In[29]:


fig,ax=plt.subplots(2,1,figsize=(16,8))

ax[0].set_title('Trend Pattern of Ngurah Rai Passengers Dataset')
ax[0].plot(data_trend,color='green')

ax[1].set_title('Cyclic Pattern of Ngurah Rai Passengers Dataset')
ax[1].plot(data_cycle,color='green');


# In[30]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize']=12,5
decompose = seasonal_decompose(data, model='mul')
decompose.plot();


# #### Simple Moving Average

# In[31]:


data = data['12_month_SMA'] = data.rolling(window=12).mean()
data.tail()


# In[33]:


plt.figure(figsize=(12,5))
plt.title('12 Month Simple Moving Average Prediction')
plt.plot(data['Passengers'],color='blue',label='Actual Data')
plt.plot(data['12_month_SMA'],color='orange',label='SMA Prediction')
plt.legend(fontsize=15);


# In[177]:


predict1 = data.rolling(window=12).mean()


# In[178]:


prediction2 = predict1.predict(start=start, end=end, dynamic=False, typ='levels')


# Failed to predict seasonality but managed to predict the trend pattern

# #### Exponentially Weighed Moving Average

# In[34]:


data1 = pd.read_excel('C:/Users/LENOVO PC/data.xlsx',index_col='Period',parse_dates=True)
data1.index.freq = 'MS'
data1.head()


# In[35]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 12
alpha = 2/(span+1)
data1['12_month_EWMA']=SimpleExpSmoothing(data1['Passengers']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
data1.head()


# In[36]:


plt.figure(figsize=(12,5))
plt.title('12 Month Exponentially Weighted Moving Average Prediction')
plt.plot(data1['Passengers'],color='green',label='Actual Data')
plt.plot(data1['12_month_EWMA'],color='red',label='EWMA Prediction')
plt.legend(fontsize=15);


# #### Data Splitting

# In[37]:


df = pd.read_excel('C:/Users/LENOVO PC/data.xlsx',index_col='Period',parse_dates=True)
df.index.freq = 'MS'
df.head()


# In[38]:


df.info()


# In[39]:


train = df.iloc[:122]
test = df.iloc[122:]


# #### Holt-Winters

# In[40]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train['Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# In[41]:


test_predictions = fitted_model.forecast(52).rename('HW Forecast')


# In[42]:


test_predictions


# In[91]:


len(test_predictions)


# In[43]:


train['Passengers'].plot(legend=True,label='TRAIN')
test['Passengers'].plot(legend=True,label='TEST',figsize=(12,9));


# In[44]:


train['Passengers'].plot(legend=True,label='TRAIN')
test['Passengers'].plot(legend=True,label='TEST',figsize=(12,9))
test_predictions.plot(legend=True,label='PREDICTION');


# In[45]:


train['Passengers'].plot(legend=True,label='TRAIN')
test['Passengers'].plot(legend=True,label='TEST',figsize=(12,9))
test_predictions.plot(legend=True,label='PREDICTION',xlim=['2016-01-01','2019-01-01']);


# In[46]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[47]:


MAE_HW = mean_absolute_error(test,test_predictions)
MAE_HW


# In[48]:


MSE_HW = mean_squared_error(test,test_predictions)
MSE_HW


# In[49]:


RMSE_HW = np.sqrt(mean_squared_error(test,test_predictions))
RMSE_HW


# In[50]:


test.describe()


# In[51]:


final_model = ExponentialSmoothing(df['Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# In[52]:


forecast_predictions = final_model.forecast(52)


# In[53]:


df['Passengers'].plot(figsize=(12,9))
forecast_predictions.plot();


# #### Autoregression

# In[54]:


from statsmodels.tsa.ar_model import AutoReg,ARResults
nlag = 3


# In[55]:


title='Monthly Passengers in Ngurah Rai Airport Forecast'
ylabel='Passengers (thousands)'
xlabel='' 

ax = df['Passengers'].plot(figsize=(12,6),title=title);
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# In[56]:


model = AutoReg(train['Passengers'], lags=nlag)


# In[57]:


AR1fit = model.fit()


# In[58]:


start=len(train)
end=len(train)+len(test)-1
predictions1 = AR1fit.predict(start=start, end=end).rename('Predictions')


# In[60]:


predictions1


# #### Plotting

# In[61]:


test['Passengers'].plot(legend=True)
predictions1.plot(legend=True,figsize=(12,9));


# In[62]:


MAE_AR = mean_absolute_error(test, predictions1)
MAE_AR


# In[63]:


MSE_AR = mean_squared_error(test,predictions1)
MSE_AR


# In[64]:


RMSE_AR = np.sqrt(MSE_AR)
RMSE_AR


# In[103]:


model = AutoReg(df['Passengers'], lags=nlag)


# In[104]:


ARfit = model.fit()


# In[105]:


fcast = ARfit.predict(start=len(df), end=len(df)+19, dynamic=False).rename('Forecast')


# In[106]:


df['Passengers'].plot(legend=True)
fcast.plot(legend=True,figsize=(12,9));


# #### Arima

# In[120]:


df.head()


# In[123]:


len(df)


# In[125]:


from statsmodels.tsa.arima.model import ARIMAResults,ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
p = 0
d = 1
q = 3


from pmdarima import auto_arima


# In[126]:


auto_arima(df['Passengers'])


# In[133]:


model = ARIMA(train['Passengers'],order=(p,d,q))
results = model.fit()
results.summary()


# In[134]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(p,d,q) Predictions')


# In[135]:


title = 'Passengers in Ngurah Rai'
ylabel='Passengers'
xlabel='' 

ax = df['Passengers'].plot(legend=True,figsize=(12,9),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[136]:


model = ARIMA(df['Passengers'],order=(p,d,q))
results = model.fit()
fcast = results.predict(len(df),len(df)+28,typ='levels').rename('ARIMA(p,d,q) Forecast')


# In[137]:


title = 'Passengers in Ngurah Rai'
ylabel='Passengers'
xlabel='' 

ax = df['Passengers'].plot(legend=True,figsize=(12,9),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[138]:


MAE_ARIMA1 = mean_absolute_error(test, predictions)
MAE_ARIMA1


# In[139]:


MSE_ARIMA1 = mean_squared_error(test,predictions)
MSE_ARIMA1


# In[140]:


RMSE_ARIMA1 = np.sqrt(MSE_ARIMA1)
RMSE_ARIMA1


# #### ARIMA Different Order

# In[155]:


model1 = ARIMA(train['Passengers'],order=(0,1,1))
results1 = model1.fit()
results1.summary()


# In[156]:


predictions11 = results1.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(0,1,1) Predictions')


# In[158]:


title = 'Passengers in Ngurah Rai'
ylabel='Passengers'
xlabel='' 

ax = test['Passengers'].plot(legend=True,figsize=(12,9),title=title)
predictions11.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[159]:


model1 = ARIMA(df['Passengers'],order=(0,1,1))
results1 = model1.fit()
fcast1 = results1.predict(len(df),len(df)+28,typ='levels').rename('ARIMA(0,1,1) Forecast')


# In[160]:


title = 'Passengers in Ngurah Rai'
ylabel='Passengers'
xlabel='' 

ax = df['Passengers'].plot(legend=True,figsize=(12,9),title=title)
fcast1.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[161]:


MAE_ARIMA2 = mean_absolute_error(test,predictions11)
MAE_ARIMA2


# In[162]:


MSE_ARIMA2 = mean_squared_error(test,predictions11)
MSE_ARIMA2


# In[163]:


RMSE_ARIMA2 = np.sqrt(MSE_ARIMA2)
RMSE_ARIMA2 


# In[164]:


#### SARIMA


# In[165]:


import matplotlib.pyplot as plt
import pmdarima as pm


# In[167]:


smodel = pm.auto_arima(df, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)


# In[168]:


smodel.summary()


# In[169]:


n_periods = 52
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(df.index[-1], periods = n_periods, freq='MS')


fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)


plt.plot(df)
plt.plot(fitted_series, color='red')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Passengers in Ngurah Rai")
plt.show()


# In[170]:


import statsmodels.api as sm
model_SARIMA = sm.tsa.statespace.SARIMAX(df['Passengers'],order=(1,1,0),seasonal_order=(0,1,2,12))
results_sarima = model_SARIMA.fit()


# In[171]:


df2 = df.copy()
df2['SARIMA Predictions']=results_sarima.predict(start=start,end=end,dynamic=True)
df2[['Passengers','SARIMA Predictions']].plot(figsize=(12,8))


# In[172]:


predictions_sarima = results_sarima.predict(start=start, end=end, dynamic=True)
predictions_sarima.plot()


# In[173]:


MAE_SARIMA = mean_absolute_error(test,predictions_sarima)
MAE_SARIMA


# In[174]:


MSE_SARIMA = mean_squared_error(test,predictions_sarima)
MSE_SARIMA


# In[175]:


RMSE_SARIMA = np.sqrt(MSE_SARIMA)
RMSE_SARIMA


# In[342]:


data_used         = ['Holt-Winters', 'Autoregressionn', 'ARIMA Order(0,1,0)', 'ARIMA Order(2,2,1)', 'SARIMA']
MAE               = [MAE_HW, MAE_AR, MAE_ARIMA1, MAE_ARIMA2, MAE_SARIMA]
MSE               = [MSE_HW, MSE_AR, MSE_ARIMA1, MSE_ARIMA2, MSE_SARIMA]
RMSE              = [RMSE_HW, RMSE_AR, RMSE_ARIMA1, RMSE_ARIMA2, RMSE_SARIMA]
model_performance = pd.DataFrame({'Data': data_used, 'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE})
model_performance


# #### Detecting Anomalies

# In[182]:


data0 = pd.read_excel('C:/Users/LENOVO PC/data.xlsx',index_col='Period',parse_dates=True)
data0.index.freq = 'MS'
data0.head()


# In[246]:


import tensorflow as tf
from tensorflow import keras
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# In[272]:


train_size = int(len(df) * 0.7)
test_size = len(df) - train_size

train, test = df.iloc[:train_size], df.iloc[train_size:]
print(train.shape, test.shape)


# In[273]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[274]:


scaler = scaler.fit(train[['Passengers']])


# In[275]:


len(train)


# In[276]:


train['Passengers'] = scaler.transform(train[['Passengers']])
test['Passengers'] = scaler.transform(test[['Passengers']])


# In[277]:


train.head()


# In[278]:


test.shape


# In[280]:


model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64, 
    input_shape=(train.shape[0], train.shape[1])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=train.shape[0]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=train.shape[1])))

model.compile(loss='mae', optimizer='adam', metrics = ['accuracy'])


# In[281]:


history = model.fit(
    train, train.Passengers,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# In[258]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();


# In[284]:


TIME_STEPS = 30
X_train, y_train = create_dataset(train[['Passengers']], train.Passengers, TIME_STEPS)
X_test, y_test = create_dataset(test[['Passengers']], test.Passengers, TIME_STEPS)

print(X_train.shape)


# In[285]:


model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64, 
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))

model.compile(loss='mae', optimizer='adam', metrics = ['accuracy'])


# In[286]:


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# In[287]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();


# In[288]:


X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)


# In[289]:


sns.distplot(train_mae_loss, bins=50, kde=True);


# In[290]:


X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)


# In[337]:


THRESHOLD = 0.678

test_score_data0 = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_data0['loss'] = test_mae_loss
test_score_data0['threshold'] = THRESHOLD
test_score_data0['anomaly'] = test_score_data0.loss > test_score_data0.threshold
test_score_data0['Passengers'] = test[TIME_STEPS:].Passengers


# In[338]:


plt.plot(test_score_data0.index, test_score_data0.loss, label='loss')
plt.plot(test_score_data0.index, test_score_data0.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();


# In[339]:


anomalies = test_score_data0[test_score_data0.anomaly == True]
anomalies.head()


# In[340]:


anomalies


# In[341]:


plt.plot(
  test[TIME_STEPS:].index, 
  scaler.inverse_transform(test[TIME_STEPS:].Passengers), 
  label='Passengers'
);

sns.scatterplot(
  anomalies.index,
  scaler.inverse_transform(anomalies.Passengers),
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();

