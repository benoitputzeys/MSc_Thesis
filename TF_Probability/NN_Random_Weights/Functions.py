import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import keras
from keras import layers
import numpy as np

# Define the posterior trainable distribution.
def posterior_mean_field(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-2 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Define the prior trainable distribution.
def prior_trainable(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the prior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),  # Returns a trainable variable of shape n, regardless of input
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

# Build the model with the same hyperparameters as the "normal" NN but only the first layer is randomly distributed.
def build_model(input_dim,learning_rate):
    model = keras.Sequential([
        tfp.layers.DenseVariational(300, activation='relu', input_dim=input_dim,
                                    make_posterior_fn = posterior_mean_field,
                                    make_prior_fn = prior_trainable),
        layers.Dense(275, activation='relu',),
        layers.Dense(250, activation='relu',),
        layers.Dense(250, activation='relu',),
        layers.Dense(75, activation='relu',),
        layers.Dense(50, activation='relu',),
        layers.Dense(1),])

    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error',
                                                      'mean_absolute_error',
                                                      'mean_absolute_percentage_error'])

    return model


