import numpy as np
import gym
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.spaces import Box
import cv2


class ResizeObservation(gym.ObservationWrapper):
    """Downsample the image observation to a square image. """
    
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""

        
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # TODO accumulate reward and repeat the same action 
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


def define_env():
  env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

  env = SkipFrame(env)
  env = GrayScaleObservation(env, keep_dim=False)
  env = ResizeObservation(env, shape=84)
  env = FrameStack(env, num_stack=4)
  env._max_episode_steps = 30000
  env = JoypadSpace(
      env,
      [
      ['NOOP'],
      ['right'],
      ['right', 'A'],
      ['right', 'B'],
      ['right', 'A', 'B'],
      ]
  )
  return env
