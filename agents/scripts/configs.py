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

"""Example configurations using the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-variable
import gym
import tensorflow as tf

from agents import ppo, tools
from agents.scripts import networks
from agents.tools.pong_debug_env import DebugPong, ObservationType


def default():
  """Default configuration for PPO."""
  # General
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30
  use_gpu = True
  # Network
  network = networks.feed_forward_gaussian
  distribution_class = networks.getMultivariateNormalDiagClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  policy_layers = 200, 100
  value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = True
  normalize_observations = True
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-4
  # Losses
  discount = 0.995
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  value_loss_coeff = 1
  policy_loss_coeff = 1
  return locals()


def simple_pong():
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30

  use_gpu = True
  # Network
  # network = networks.feed_forward_gaussian
  network = networks.feed_forward_categorical
  # distribution_class = networks.getMultivariateNormalDiagClass
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  policy_layers = 200, 100
  value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = False
  normalize_observations = False
  init_logstd = -1
  # Optimization
  update_every = 60
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-4
  # Losses
  discount = 0.985
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  value_loss_coeff = 1
  policy_loss_coeff = 1
  env = lambda: DebugPong(gym.make("Pong-v0"))
  max_length = 1000
  steps = 20e6  # 20M

  return locals()

def simple_cnn_pong():
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30

  use_gpu = True
  # Network
  # network = networks.feed_forward_gaussian
  network = networks.feed_forward_cnn_small_categorical
  # distribution_class = networks.getMultivariateNormalDiagClass
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  # policy_layers = 200, 100
  # value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = False
  normalize_observations = False
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-4
  # Losses
  discount = 0.985
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  value_loss_coeff = 1
  policy_loss_coeff = 1
  env = lambda: DebugPong(gym.make("Pong-v0"), observation_type=ObservationType.GRAY_FRAMES_DIFF, scale=2)
  max_length = 200
  steps = 20e6  # 20M

  return locals()

def simple_cnn_pong_clipping():
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30

  use_gpu = True
  # Network
  # network = networks.feed_forward_gaussian
  network = networks.feed_forward_cnn_small_categorical
  # distribution_class = networks.getMultivariateNormalDiagClass
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  # policy_layers = 200, 100
  # value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = False
  normalize_observations = False
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-4
  # Losses
  discount = 0.985
  clipping_coef = 0.2
  kl_init_penalty = 0

  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  value_loss_coeff = 1
  policy_loss_coeff = 1
  env = lambda: DebugPong(gym.make("Pong-v0"), observation_type=ObservationType.GRAY_FRAMES_DIFF, scale=2)
  max_length = 200
  steps = 20e6  # 20M

  return locals()

def simple_cnn_pong_two_frames():
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30

  use_gpu = True
  # Network
  # network = networks.feed_forward_gaussian
  network = networks.feed_forward_cnn_small_categorical
  # distribution_class = networks.getMultivariateNormalDiagClass
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  # policy_layers = 200, 100
  # value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = False
  normalize_observations = False
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-4
  # Losses
  discount = 0.985
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  value_loss_coeff = 1
  policy_loss_coeff = 1
  env = lambda: DebugPong(gym.make("Pong-v0"), observation_type=ObservationType.GRAY_FRAMES, scale=2)
  max_length = 200
  steps = 20e6  # 20M

  return locals()

def simple_cnn_pong_three_frames():
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30

  use_gpu = True
  # Network
  # network = networks.feed_forward_gaussian
  network = networks.feed_forward_cnn_small_categorical
  # distribution_class = networks.getMultivariateNormalDiagClass
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  # policy_layers = 200, 100
  # value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = False
  normalize_observations = False
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 2e-5
  # Losses
  discount = 0.985
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  value_loss_coeff = 1
  policy_loss_coeff = 1
  env = lambda: DebugPong(gym.make("Pong-v0"), observation_type=ObservationType.GRAY_FRAMES_BOTH, scale=2)
  max_length = 200
  steps = 20e6  # 20M

  return locals()


def simple_cnn_pong_three_frames_small_rl_cliping():
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30

  use_gpu = True
  # Network
  # network = networks.feed_forward_gaussian
  network = networks.feed_forward_cnn_small_categorical
  # distribution_class = networks.getMultivariateNormalDiagClass
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  # policy_layers = 200, 100
  # value_layers = 200, 100
  init_mean_factor = 0.1
  continuous_preprocessing = False
  normalize_observations = False
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 4e-5
  # Losses
  discount = 0.985
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  entropy_reward = 0.02
  clipping_coef = 0.2
  kl_init_penalty = 0
  value_loss_coeff = 1
  policy_loss_coeff = 1
  env = lambda: DebugPong(gym.make("Pong-v0"), observation_type=ObservationType.GRAY_FRAMES_BOTH, scale=2)
  max_length = 200
  steps = 20e6  # 20M

  return locals()

def default_atari():
  """Default configuration for PPO."""
  # General
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  continuous_preprocessing = False
  normalize_observations = False
  eval_episodes = 30
  use_gpu = False
  # Network
  network = networks.feed_forward_cnn_categorical
  distribution_class = networks.getCategoricalClass
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  # Optimization
  update_every = 10
  update_epochs = 24
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-3
  # Losses
  discount = 0.995
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  return locals()



def pong():
  locals().update(default_atari())
  use_gpu = False
  num_agents = 5
  update_every = 10
  update_epochs = 4
  learning_rate = 1e-3
  value_loss_coeff = 1
  policy_loss_coeff = 1
  gae_lambda = True
  gae_lambda = 0.95
  # Environment
  env = lambda: tools.wrappers.FrameHistory(tools.wrappers.Shrink(gym.make("Pong-v0")), past_indices=range(0, 2), flatten=True)
  max_length = 300
  steps = 2e6  # 2M
  return locals()

def pong_16():
  """Configuration for the pendulum classic control task."""
  locals().update(pong())
  policy_loss_coeff = 16
  return locals()

def pong_32():
  """Configuration for the pendulum classic control task."""
  locals().update(pong())
  policy_loss_coeff = 32
  return locals()

def pong_64():
  """Configuration for the pendulum classic control task."""
  locals().update(pong())
  policy_loss_coeff = 64
  return locals()


def pong_128():
  """Configuration for the pendulum classic control task."""
  locals().update(pong())
  policy_loss_coeff = 128
  return locals()


def pendulum():
  """Configuration for the pendulum classic control task."""
  locals().update(default())
  # Environment
  env = 'Pendulum-v0'
  max_length = 200
  steps = 2e6  # 2M
  return locals()

def pong_continuous():
  """Configuration for MuJoCo's reacher task."""
  locals().update(default())
  # Environment
  env = lambda: DebugPong(gym.make("Pong-v0"), discrete_control=False)
  max_length = 1000
  use_gpu = False
  steps = 5e6  # 5M
  discount = 0.985
  update_every = 60
  return locals()


def reacher():
  """Configuration for MuJoCo's reacher task."""
  locals().update(default())
  # Environment
  env = 'Reacher-v1'
  max_length = 1000
  steps = 5e6  # 5M
  discount = 0.985
  update_every = 60
  return locals()


def cheetah():
  """Configuration for MuJoCo's half cheetah task."""
  locals().update(default())
  # Environment
  env = 'HalfCheetah-v1'
  max_length = 1000
  steps = 1e7  # 10M
  discount = 0.99
  return locals()


def walker():
  """Configuration for MuJoCo's walker task."""
  locals().update(default())
  # Environment
  env = 'Walker2d-v1'
  max_length = 1000
  steps = 1e7  # 10M
  return locals()


def hopper():
  """Configuration for MuJoCo's hopper task."""
  locals().update(default())
  # Environment
  env = 'Hopper-v1'
  max_length = 1000
  steps = 1e7  # 10M
  update_every = 60
  return locals()


def ant():
  """Configuration for MuJoCo's ant task."""
  locals().update(default())
  # Environment
  env = 'Ant-v1'
  max_length = 1000
  steps = 2e7  # 20M
  return locals()


def humanoid():
  """Configuration for MuJoCo's humanoid task."""
  locals().update(default())
  # Environment
  env = 'Humanoid-v1'
  max_length = 1000
  steps = 5e7  # 50M
  update_every = 60
  return locals()
