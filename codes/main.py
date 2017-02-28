import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AR
import data_preparation

# Load dataset
data_set = data_preparation.read_data('./data_set/HourlyDemands_2002-2016.csv')
data, label = data_preparation.split_label(data_set, 'Ontario Demand')
print('Data set Loaded!')
print(data.shape)
print(label.shape)

# Splitting train and test data
train_data, test_data = data[0:119832], data[119832:]
train_label, test_label = label[0:119832], label[119832:]

# Implementing Models
df = pd.concat([label.shift(48), label], axis=1)
df.columns = ['t-1', 't+1']
X = df.values
train, test = X[0:119832], X[119832:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]

# Mean
years = []
for i in range(0, 365 * 24):
    temp_mean = train_label[i]
    for j in range(1, 11):
        temp_mean += train_label[i + (j * 365 * 24)]
    temp_mean /= 11
    years.append(temp_mean)

plt.plot(np.linspace(0, 365 * 24, 365 * 24), years, color='red', alpha=0.9)
plt.show()
print 1.0 / len(years) * sum(abs(years - test_label[:8760].values))
# Implementing Auto Regression Model
# Training Autoregression

df = pd.DataFrame((years[1:40 * 24 + 1])).to_csv('predictions.csv', index=False)

print(train[:, 1])
model = AR(train_y)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

# Making predictions
predictions = model_fit.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1, dynamic=False)
error = mean_absolute_error(test_y, predictions)
print('Test MAE: %.3f' % error)

# Plotting results
plt.plot(test_y, label="real values")
plt.plot(predictions, color='red', label="predictions", alpha=0.8)
plt.legend(loc='upper left')

plt.show()

# Making predictions for 1st Jan to 10th Feb 2017
model = AR(X[:, 1])
model_fit = model.fit()
predictions = model_fit.predict(start=len(X), end=(len(X) + 960), dynamic=False )

# Plotting results
plt.plot(predictions, color='blue', label="predictions")
plt.legend(loc='upper left')
plt.show()
