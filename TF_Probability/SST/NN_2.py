from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from TF_Probability.Functions import build_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import keras
from keras import layers
import numpy as np

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
X = X[:48*8,:]

y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))
y = y[:48*8]

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
# X_train = x_scaler.fit_transform(X_train)
# X_test = x_scaler.transform(X_test)
# y_train = y_scaler.fit_transform(y_train)

n_epochs = 5000
n_samples = X_train.shape[0]
n_batches = 10
batch_size = np.floor(n_samples/n_batches)
buffer_size = n_samples

# Build the model.

def posterior_mean_field(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the prior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),  # Returns a trainable variable of shape n, regardless of input
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def build_model():
    model = keras.Sequential([
        tfp.layers.DenseVariational(128, activation='relu',
                                input_dim=len(X_train[1]),
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable),
        layers.Dense(128, activation='relu'),
        layers.Dense(1),
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

model = build_model()

# Run training session.
model.fit(X_train,y_train, epochs=n_epochs, verbose=False)
# Describe model.
model.summary()

# Plot the learning progress.
plt.plot(model.history.history["mae"])
plt.show()

# Make a single prediction on the test set and plot.
prediction = model.predict(X_test)

fig1 = plt
fig1.plot(prediction, label = "prediction", color = "blue")
fig1.plot(y_test, label = "true", color = "red")
fig1.xlabel('Settlement Periods (Test Set)')
fig1.ylabel('Load [MW]')
fig1.legend()
fig1.show()

# Make a 100 predictions on the test set and plot.
yhats = [model.predict(X_test) for _ in range(100)]
predictions = np.array(yhats)
predictions = predictions.reshape(-1,len(predictions[1]))
predictions_vector = predictions.reshape(-1, 1)

predictions_and_errors = np.zeros((len(predictions_vector),2))
predictions_and_errors[:,:-1] = predictions_vector

j=0
for i in range (len(predictions_vector)):
        predictions_and_errors[i,1] = predictions_vector[i]-y_test[j]
        j=j+1
        if j == len(y_test):
            j=0

fig2=plt
fig2.plot(predictions[1,:].T,label = "prediction", alpha = 0.1, color = "blue")
fig2.plot(predictions[1:,:].T, alpha = 0.1, color = "blue")
fig2.plot(y_test, label = "true", alpha = 1, color = "red")
fig2.xlabel('Settlement Periods (Test Set)')
fig2.ylabel('Load [MW]')
fig2.legend()
fig2.show()

# Plot the histogram of the errors.
plt.hist(predictions_and_errors[:,1], bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()

plt.plot(X_train[:,0])
plt.xlabel("SP")
plt.ylabel("Load [MW]")
plt.show()