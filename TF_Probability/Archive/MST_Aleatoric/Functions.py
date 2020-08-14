import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
from tensorflow_probability import distributions as tfd

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

def build_model(learning_rate):
    model = keras.Sequential([
        tfp.layers.DenseVariational(128, activation='relu', input_dim=2),
        # tfp.layers.DenseVariational(64, activation='relu',
        #                             make_posterior_fn=posterior_mean_field,
        #                             make_prior_fn=prior_trainable),
        layers.Dense(128, activation='relu'),
        layers.Dense(2),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[...,0], scale=1e-3 + tf.math.softplus(0.01*t[...,1:]))),
    ])

    opt = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return model
