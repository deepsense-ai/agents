# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions for the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import operator


import tensorflow as tf
import math
from abc import ABC, abstractmethod

class Distribution(ABC):

    @abstractmethod
    def distribution_params_shape(self):
        raise NotImplementedError()

    def sample(self):
        return self._dist.sample()

    @abstractmethod
    def logpdf(self, x):
        raise NotImplementedError()

    @abstractmethod
    def max_likelihood(self):
        raise NotImplementedError()

    @abstractmethod
    def entropy(self):
        raise NotImplementedError()

def getMultivariateNormalDiagClass(): #TODO: Is it possible to do this better?
    return MultivariateNormalDiagDistribution

class MultivariateNormalDiagDistribution(Distribution):
    """
    Wraps tf.contrib.distributions.MultivariateNormalDiagDistribution with methods used by PPO
    """
    def __init__(self, params):
        self._mean, self._logstd = tf.split(params, 2, axis=2)
        self._dist = tf.contrib.distributions.MultivariateNormalDiag(self._mean, tf.exp(self._logstd))
        # self._action_size = action_size

    @staticmethod
    def distribution_params_shape(action_shape):
        return (action_shape[:-1]) + (2*action_shape[-1],)

    def max_likelihood(self):
        return self._mean

    def logpdf(self, x):
        constant = -0.5 * math.log(2 * math.pi) - self._logstd
        value = -0.5 * ((x - self._mean) / tf.exp(self._logstd)) ** 2
        return tf.reduce_sum(constant + value, -1)

    def entropy(self):
        """Empirical entropy of a normal with diagonal covariance."""
        constant = self._mean.shape[-1].value * math.log(2 * math.pi * math.e)
        return (constant + tf.reduce_sum(2 * self._logstd, 1)) / 2

    def kl(self, other_params):
        mean1, logstd1 = tf.split(other_params, 2, axis=2)
        logstd0_2, logstd1_2 = 2 * self._logstd, 2 * logstd1
        return 0.5 * (
            tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
            tf.reduce_sum((mean1 - self._mean) ** 2 / tf.exp(logstd1_2), -1) +
            tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
            self._mean.shape[-1].value)


def getCategoricalClass():
  return CategoricalDistibution

class CategoricalDistibution(Distribution):

    def __init__(self, params):
        self._logits = params
        self._dist = tf.contrib.distributions.Categorical(logits = self._logits)
        # self._action_size = action_size

    @staticmethod
    def distribution_params_shape(action_shape):
        return action_shape

    def max_likelihood(self):
        max_l = tf.argmax(self._logits, axis=2, output_type=tf.int32)
        # max_l = tf.Print(max_l, [max_l], "max_l=")

        return max_l

    def logpdf(self, x):
        return self._dist.log_prob(x)

    def entropy(self):
        return self._dist.entropy()

    def kl(self, other_params):
        other_dist = tf.contrib.distributions.Categorical(other_params)
        return tf.distributions.kl_divergence(self._dist, other_dist)

NetworkOutput = collections.namedtuple(
    'NetworkOutput', 'value, state, policy, distribution_params')


def feed_forward_gaussian(
    config, action_shape, observations, unused_length, state=None):
  """Independent feed forward networks for policy and value.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector.

  Args:
    config: Configuration object.
    action_size: Length of the action vector.
    observations: Sequences of observations.
    unused_length: Batch of sequence lengths.
    state: Batch of initial recurrent states.

  Returns:
    NetworkOutput tuple.
  """
  mean_weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_mean_factor)
  logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])

  with tf.variable_scope('policy'):
    x = flat_observations
    for size in config.policy_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    mean = tf.contrib.layers.fully_connected(
        x, action_shape[1], tf.tanh,
        weights_initializer=mean_weights_initializer)
    logstd = tf.get_variable(
        'logstd', mean.shape[2:], tf.float32, logstd_initializer)
    logstd = tf.tile(
        logstd[None, None],
        [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, 'mean')
  logstd = tf.check_numerics(logstd, 'logstd')
  value = tf.check_numerics(value, 'value')
  distribution_params = tf.concat([mean, logstd], axis=2)
  policy = MultivariateNormalDiagDistribution(distribution_params)

#  value = tf.Print(value, [tf.shape(value)], "value shape=")

  return NetworkOutput(value, state, policy, distribution_params)


def feed_forward_categorical(
    config, action_shape, observations, unused_length, state=None):

  mean_weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_mean_factor)

  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  flat_observations = tf.to_float(flat_observations)
  flat_observations = (flat_observations - 50)*0.01
  # flat_observations = tf.Print(flat_observations, [flat_observations], "flat_observations=")
  with tf.variable_scope('policy'):
    x = flat_observations

    for size in config.policy_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(
        x, action_shape[1], tf.tanh,
        weights_initializer=mean_weights_initializer)
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]

  # value = tf.Print(value, [tf.shape(value)], "value shape=")
  #
  # value = tf.Print(value, [tf.value], "value=")
  # logits = 0.1*logits
  # logits = tf.Print(logits, [logits], "logits shape=")


  policy = CategoricalDistibution(logits)

  return NetworkOutput(value, state, policy, logits)



def feed_forward_cnn_small_categorical(
    config, action_shape, observations, unused_length, state=None):

  # delta = observations[..., 2] - (observations[...,0]  - observations[..., 1])
  # test = tf.linalg.norm(tf.to_float(delta))
  # observations = tf.Print(observations, [test], "Sanity test=")

  # observations = observations[..., None]
  observations = observations[...,:2]
  obs_shape = observations.shape.as_list()
  x = tf.reshape(observations, [-1]+ obs_shape[2:])

  with tf.variable_scope('policy'):
    x = tf.to_float(x)/255.0
    x = tf.contrib.layers.conv2d(x, 32, [5, 5], [2, 2], activation_fn= tf.nn.relu, padding="SAME")
    x = tf.contrib.layers.conv2d(x, 32, [5, 5], [2, 2], activation_fn=tf.nn.relu, padding="SAME")

    flat_x = tf.reshape(x, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])

    # flat_x = tf.reshape(x, [obs_shape[0], obs_shape[1],
    #   functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])

    x = tf.contrib.layers.fully_connected(flat_x, 128, tf.nn.relu)

    logits = tf.contrib.layers.fully_connected(x, action_shape[1], activation_fn=None)

    value = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)[..., 0]

    # logits_cast = logits[0, 0, ...]
    # logits_cast = tf.Print(logits_cast, [tf.shape(logits_cast)], "logits_cast shape=")

    # logits = tf.Print(logits, [logits_cast], "logits_1=", summarize=12)
    # logits = tf.Print(logits, [logits[1,]], "logits_2=")

  policy = CategoricalDistibution(logits)

  return NetworkOutput(value, state, policy, logits)


def feed_forward_cnn_categorical(
    config, action_shape, observations, unused_length, state=None):

  x = tf.reshape(observations, [-1]+ observations.shape.as_list()[2:])

  with tf.variable_scope('policy'):
    x = tf.to_float(x)/255.0
    x = tf.contrib.layers.conv2d(x, 32, [5, 5], [1, 1], activation_fn= tf.nn.relu, padding="SAME")
    x = tf.contrib.layers.max_pool2d(x, [2, 2], padding="VALID")
    x = tf.contrib.layers.conv2d(x, 32, [5, 5], [1, 1], activation_fn=tf.nn.relu, padding="SAME")
    x = tf.contrib.layers.max_pool2d(x, [2, 2], padding="VALID")
    x = tf.contrib.layers.conv2d(x, 64, [4, 4], [1, 1], activation_fn=tf.nn.relu, padding="SAME")
    x = tf.contrib.layers.max_pool2d(x, [2, 2], padding="VALID")
    x = tf.contrib.layers.conv2d(x, 64, [3, 3], [1, 1], activation_fn=tf.nn.relu, padding="SAME")

    flat_x = tf.reshape(x, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])

    x = tf.contrib.layers.fully_connected(flat_x, 128, tf.nn.relu)

    logits = tf.contrib.layers.fully_connected(x, action_shape[1], activation_fn=None)

    value = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)[..., 0]

    # logits_cast = logits[0, 0, ...]
    # logits_cast = tf.Print(logits_cast, [tf.shape(logits_cast)], "logits_cast shape=")

    # logits = tf.Print(logits, [logits_cast], "logits_1=", summarize=12)
    # logits = tf.Print(logits, [logits[1,]], "logits_2=")

  policy = CategoricalDistibution(logits)

  return NetworkOutput(value, state, policy, logits)


def recurrent_gaussian(
    config, action_size, observations, length, state=None):
  """Independent recurrent policy and feed forward value networks.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector. The last policy layer is
  recurrent and uses a GRU cell.

  Args:
    config: Configuration object.
    action_size: Length of the action vector.
    observations: Sequences of observations.
    length: Batch of sequence lengths.
    state: Batch of initial recurrent states.

  Returns:
    NetworkOutput tuple.
  """
  mean_weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_mean_factor)
  logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)
  cell = tf.contrib.rnn.GRUBlockCell(config.policy_layers[-1])
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  with tf.variable_scope('policy'):
    x = flat_observations
    for size in config.policy_layers[:-1]:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    x, state = tf.nn.dynamic_rnn(cell, x, length, state, tf.float32)
    mean = tf.contrib.layers.fully_connected(
        x, action_size, tf.tanh,
        weights_initializer=mean_weights_initializer)
    logstd = tf.get_variable(
        'logstd', mean.shape[2:], tf.float32, logstd_initializer)
    logstd = tf.tile(
        logstd[None, None],
        [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, 'mean')
  logstd = tf.check_numerics(logstd, 'logstd')
  value = tf.check_numerics(value, 'value')
  policy = tf.contrib.distributions.MultivariateNormalDiag(
      mean, tf.exp(logstd))
  # assert state.shape.as_list()[0] is not None
  return NetworkOutput(policy, mean, logstd, value, state)
