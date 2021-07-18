import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers 

import logging
tf.get_logger().setLevel(logging.ERROR)

class Envelope(layers.Layer):
    """
    Envelope function that ensures a smooth cutoff
    """
    def __init__(self, exponent, name='envelope', **kwargs):
        super().__init__(name=name, **kwargs)
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def call(self, inputs):

        # Envelope function divided by r
        env_val = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)

        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))


class DistanceLayer(layers.Layer):
    """
    Projects a distance 0 < r < r_cut into an orthogonal basis of Bessel functions
    """
    def __init__(self, num_radial=8, r_cut=5.0, envelope_exponent=5,
                 name='bessel_basis', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_radial = num_radial
        self.inv_cutoff = tf.constant(1 / r_cut, dtype=tf.float32)
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        def freq_init(shape, dtype):
            return tf.constant(np.pi * np.arange(1, shape + 1, dtype=np.float32), dtype=dtype)
        self.frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
                                           dtype=tf.float32, initializer=freq_init, trainable=True)

    def call(self, inputs):
        # scale to range [0, 1]
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = tf.expand_dims(d_scaled, -1)

        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * tf.sin(self.frequencies * d_scaled)

class FeedForwardLayer(layers.Layer):
    """
    Convenience layer for defining a feed-forward neural network (a number of sequential dense layers)
    """
    def __init__(self, layer_sizes, layer_activations, name, **kwargs):
        super().__init__(name=name, **kwargs)

        n_layers = len(layer_sizes)
        assert n_layers == len(layer_activations)

        self.layer_list = [layers.Dense(layer_sizes[i], activation=layer_activations[i]) for i in range(n_layers)]


    def call(self, inputs):

        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

class SimpleDistanceLayer(layers.Layer):
    def __init__(self, r_cut=5.0, name='distance_layer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.r_cut = r_cut

    def call(self, dR):

        oodR = tf.math.reciprocal(dR)
        cosdR = (tf.math.cos(dR * math.pi / self.r_cut) + 1.0) / 2.0
        output = tf.stack([dR, oodR, cosdR], axis=1)
        return output
