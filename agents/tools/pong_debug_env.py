from collections import deque

import gym
from gym import Wrapper
from gym.spaces import Discrete, Box
import numpy as np
from enum import Enum


class ObservationType(Enum):
  STATE = 1
  GRAY_FRAMES_DIFF = 2
  GRAY_FRAMES = 3
  GRAY_FRAMES_BOTH = 4


class DebugPong(Wrapper):

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, env:gym.Env, scale = 1, observation_type=ObservationType.STATE,
               discrete_control=True, random_pong=False):
    self._random_pong = random_pong
    self._discrete_control = discrete_control
    if discrete_control:
      self.action_space = Discrete(2)
    else:
      self.action_space = Box(-10, 10, shape=(1,))
    self.env = env
    self.orignal_get_image = self.env.env._get_image
    self.env.env._get_image = self.hacked_get_image
    self._buffer_states = deque([], maxlen=2)
    self._buffer_frames = deque([], maxlen=2)
    self._scale = scale
    self._observation_type = observation_type
    if self._observation_type == ObservationType.STATE:
      self.observation_space = Box(0, 255, shape=(8,))
    if self._observation_type == ObservationType.GRAY_FRAMES_DIFF:
      self.observation_space = Box(-255, 255, shape=(int(152/self._scale), int(128/self._scale), 1))
    if self._observation_type == ObservationType.GRAY_FRAMES:
      self.observation_space = Box(-255, 255, shape=(int(152/self._scale), int(128/self._scale), 2))
    if self._observation_type == ObservationType.GRAY_FRAMES_BOTH:
      self.observation_space = Box(-255, 255, shape=(int(152/self._scale), int(128/self._scale), 3))


  def hacked_get_image(self):
    image_obs = self._get_image_obs(scale = self._scale)
    new_image = np.stack((image_obs, image_obs, image_obs), axis=2)
    return new_image

  def _get_image_obs(self, scale = 1):
    return self.orignal_get_image()[40:192:scale, 16:144:scale, 0]

  def find_ball(self, image, default=None):
    ball_area = image[40:193, 20:140, 0]
    res = np.argwhere(ball_area==236)
    if len(res)==0:
      return default
    else:
      return res[0]

  def find_agent(self, image, default=None):
    agent_area = image[40:193, 140, 0]
    # print(np.unique(agent_area))
    res = np.argwhere(agent_area==92)
    if len(res)==0:
      return default
    else:
      return res[0]

  def find_oponent(self, image, default=None):
    agent_area = image[40:193, 19, 0]
    # print(np.unique(agent_area))
    res = np.argwhere(agent_area==213)
    if len(res)==0:
      return default
    else:
      return res[0]


  def _step(self, action):
    if self._discrete_control:
      if action==0:
        original_action = 2
      else:
        original_action = 3
    else: #continuous control
      if action>0:
        original_action = 2
      else:
        original_action = 3

    if self._random_pong:
      import random
      x = random.uniform(0, 1)
      if x>0.5:
        original_action = 2
      else:
        original_action = 3


    _, rew, done, info = self.env._step(original_action)
    self._update_buffers()
    obs = self._get_obs()

    return obs, rew, done, info

  def _update_buffers(self):
    rgb_obs = self._get_image_obs(self._scale)
    self._buffer_frames.append(rgb_obs)
    original_rgb_image = self.orignal_get_image()
    state_obs = np.concatenate((self.find_oponent(original_rgb_image, [49]),
                          self.find_agent(original_rgb_image, [50]),
                          self.find_ball(original_rgb_image, [50, 50])))
    self._buffer_states.append(state_obs)

  def _reset(self, **kwargs):
    self.env.reset()
    self._update_buffers()
    self._update_buffers()
    return self._get_obs()

  def _get_obs(self):
    if self._observation_type == ObservationType.STATE:
      return np.concatenate([self._buffer_states[i] for i in range(len(self._buffer_states))], axis=-1)

    if self._observation_type == ObservationType.GRAY_FRAMES_DIFF:
      res = self._buffer_frames[0].astype('int32')- self._buffer_frames[1].astype('int32')
      res = res[...,None]
      return res

    if self._observation_type == ObservationType.GRAY_FRAMES:
      res = np.stack([self._buffer_frames[0], self._buffer_frames[1]], axis=2)

      return res.astype('int32')

    if self._observation_type == ObservationType.GRAY_FRAMES_BOTH:
      f0 = self._buffer_frames[0].astype('int32')
      f1 = self._buffer_frames[1].astype('int32')
      diff = f0 - f1
      res = np.stack([f0, f1, diff], axis=2)

      return res


        # env = gym.make('Pong-v0')
# env = DebugPong(env, return_state=False, scale=2)
# env.reset()
# for _ in range(1000):
#     # env.render()
#     env.step(env.action_space.sample())
#     # print(env.step(env.action_space.sample())) # take a random action
#     time.sleep(0.1)