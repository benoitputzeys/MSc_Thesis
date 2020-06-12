import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# Import the timeseries data
df = pd.read_csv("/Users/benoitputzeys/Desktop/Master Thesis/Data/SPI_2020/SPI_202005.csv")
print(df.head())
df_label = df["Total_Generation"]
df_features = pd.DataFrame()
df_features["Total_Generation"] = df["Total_Generation"].shift(-2)
df_features["Settlement_Period"] = df["Settlement_Period"]

# Create your input variable
x = df_features.values
y = df_label.values
y = np.reshape(y,(len(y),1))

# After having shifted the data, the nan values have to be replaces in order to have good predicitons.
replace_nan = SimpleImputer(missing_values = np.nan, strategy='mean')
replace_nan.fit(x[:,0:1])
x[:, 0:1] = replace_nan.transform(x[:,0:1])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


# Fit the SVR to our data
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Compute the prediction and rescale
intermediate_result = regressor.predict(X_test)
#print(intermediate_result)
result = y_scaler.inverse_transform(intermediate_result)
#print(result)
result = result.reshape((len(result), 1))

# Visualising the results
X_vals = []
for i in range(len(df)):
    X_vals = np.append(X_vals, i)
X_vals = np.reshape(X_vals,(len(X_vals),1))


X_grid = np.arange(1, 3408, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_vals, y, color = 'red')
plt.plot(X_vals[-682:], result, color = 'blue')
plt.title('SVR')
plt.xlabel('Settlement Period')
plt.ylabel('Prediction')
plt.show()