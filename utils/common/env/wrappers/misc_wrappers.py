'''
Taken from:
https://github.com/google-research/batch-ppo
and other random sources
'''
from collections import deque
import numpy as np
import torch
import gym
import gym.spaces
import cv2
cv2.ocl.setUseOpenCL(False)

from utils.gym_utils import LazyFrames

from griddly import GymWrapper

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class AutoReset(object):
  """Automatically reset environment when the episode is done."""

  def __init__(self, env):
    self._env = env
    self._done = True

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._done:
      observ, reward, done, info = self._env.reset(), 0.0, False, {}
    else:
      observ, reward, done, info = self._env.step(action)
    self._done = done
    return observ, reward, done, info

  def reset(self):
    self._done = False
    return self._env.reset()


class FrameDelta(object):
  """Convert the observation to a difference from the previous observation."""

  def __init__(self, env):
    self._env = env
    self._last = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    low = self._env.observation_space.low
    high = self._env.observation_space.high
    low, high = low - high, high - low
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    delta = observ - self._last
    self._last = observ
    return delta, reward, done, info

  def reset(self):
    observ = self._env.reset()
    self._last = observ
    return observ


class LimitDuration(object):
  """End episodes after specified number of steps."""

  def __init__(self, env, duration, termination_reward=None):
    self._env = env
    self._duration = duration
    self._step = None
    self._term_rew = termination_reward

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
      info['TerminationType'] = 'timeout'
      if self._term_rew:
          reward = self._term_rew
    return observ, reward, done, info

  def reset(self, **kwargs):
    self._step = 0
    return self._env.reset(**kwargs)


class OneHotAction(object):

  def __init__(self, env):
      assert isinstance(env.action_space, gym.spaces.Discrete)
      self._env = env

  def __getattr__(self, name):
      return getattr(self._env, name)

  @property
  def action_space(self):
      shape = (self._env.action_space.n,)
      space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
      space.sample = self._sample_action
      return space

  def step(self, action):
      index = np.argmax(action).astype(int)
      reference = np.zeros_like(action)
      reference[index] = 1
      if not np.allclose(reference, action):
        raise ValueError(f'Invalid one-hot action:\n{action}')
      return self._env.step(index)

  def reset(self):
      return self._env.reset()

  def _sample_action(self):
      actions = self._env.action_space.n
      index = self._random.randint(0, actions)
      reference = np.zeros(actions, dtype=np.float32)
      reference[index] = 1.0
      return reference


class ConvertTo32Bit(object):
    """Convert data types of an OpenAI Gym environment to 32 bit."""

    def __init__(self, env):
        """Convert data types of an OpenAI Gym environment to 32 bit.
        Args:
          env: OpenAI Gym environment.
        """
        self._env = env

    def __getattr__(self, name):
        """Forward unimplemented attributes to the original environment.
        Args:
          name: Attribute that was accessed.
        Returns:
          Value behind the attribute name in the wrapped environment.
        """
        return getattr(self._env, name)

    def step(self, action):
        """Forward action to the wrapped environment.
        Args:
          action: Action to apply to the environment.
        Raises:
          ValueError: Invalid action.
        Returns:
          Converted observation, converted reward, done flag, and info object.
        """
        observ, reward, done, info = self._env.step(action)
        observ = self._convert_observ(observ)
        reward = self._convert_reward(reward)
        return observ, reward, done, info

    def reset(self):
        """Reset the environment and convert the resulting observation.
        Returns:
          Converted observation.
        """
        observ = self._env.reset()
        observ = self._convert_observ(observ)
        return observ

    def _convert_observ(self, observ):
        """Convert the observation to 32 bits.
        Args:
          observ: Numpy observation.
        Raises:
          ValueError: Observation contains infinite values.
        Returns:
          Numpy observation with 32-bit data type.
        """
        if not np.isfinite(observ).all():
            raise ValueError('Infinite observation encountered.')
        if observ.dtype == np.float64:
            return observ.astype(np.float32)
        if observ.dtype == np.int64:
            return observ.astype(np.int32)
        return observ

    def _convert_reward(self, reward):
        """Convert the reward to 32 bits.
        Args:
          reward: Numpy reward.
        Raises:
          ValueError: Rewards contain infinite values.
        Returns:
          Numpy reward with 32-bit data type.
        """
        if not np.isfinite(reward).all():
            raise ValueError('Infinite reward encountered.')
        return np.array(reward, dtype=np.float32)

'''Retro wrapper'''
class Downsample(gym.ObservationWrapper):
    def __init__(self, env, ratio):
        """
        Downsample images by a factor of ratio
        """
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (oldh//ratio, oldw//ratio, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame


################### F R A M E   S T A C K I N G ################################################
''' another version of frame stack '''
class StackWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps=4, dtype=np.float32):
        super(StackWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class StackWrapperPyTorch(gym.Wrapper):
    def __init__(self, env, n_frames=4, dim_order='pytorch'):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(StackWrapperPyTorch, self).__init__(env)
        self.dim_order = dim_order
        if dim_order == 'tensorflow':
            height, width, n_channels = env.observation_space.shape
            obs_shape = [height, width, n_channels * n_frames]
        elif dim_order == 'pytorch':
            n_channels, height, width = env.observation_space.shape
            obs_shape = [n_channels * n_frames, height, width]
        else:
            raise ValueError('dim_order should be "tensorflow" or "pytorch", got {}'.format(dim_order))
        self.observation_space = gym.spaces.Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        if self.dim_order == 'tensorflow':
            offset = self.env.observation_space.shape[-1]
            axis = -1
            cropped_framebuffer = self.framebuffer[:, :, :-offset]
        elif self.dim_order == 'pytorch':
            offset = self.env.observation_space.shape[0]
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)


class FrameStackPyTorch(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        assert env.observation_space.dtype == np.uint8

        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(shp[0] * n_frames, shp[1], shp[2]),
            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))
##############################################################################


''' uses colorimetric grayscale conversion, crops top and bottoma and resizes the screen '''
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)



class ImageToPyTorch2(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)



class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape [height, width, channels]
    to Pytorch [channels, height, width]
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


class PlaNetObservations(gym.ObservationWrapper):
    ''' transforms obs from 0-255 to -0.5-0.5 scale
    returns a torch tensor or a numpy vector '''
    def __init__(self, env, ret_np_or_torch='torch'):
        super(PlaNetObservations, self).__init__(env)
        self.np_or_torch = ret_np_or_torch

    # Preprocesses an observation from [0, 255] to [-0.5, 0.5])
    def img_to_obs(self, obs, bits=5):
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.div_(2 ** (8 - bits)).floor_().div_(2 ** bits).sub_(0.5)
        obs = obs.add_(torch.rand_like(obs).div_(2 ** bits))
        if self.np_or_torch == 'np':
            return obs.detach().numpy()
        elif self.np_or_torch == 'torch':
            return obs.unsqueeze(0)

    def observation(self, obs):
        return self.img_to_obs(obs)

class RescaleFrame(gym.ObservationWrapper):
    def __init__(self, env, frame_size=(84, 84), grayscale=False, shape_type_input='chw'):
        """
        Rescale frames to 84x84 as done in the Nature paper and later work.
        """
        super().__init__(env)
        self._width, self._height = frame_size
        self._grayscale = grayscale
        self.shape_type_input = shape_type_input

        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        if shape_type_input == 'chw': # pytorch
            new_space = gym.spaces.Box(low=0, high=255, shape=(num_colors, self._height, self._width), dtype=np.uint8,)
        elif shape_type_input == 'hwc': # tf
            new_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, num_colors), dtype=np.uint8,)

        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, frame):
        if self.shape_type_input == 'chw':
            frame = np.transpose(frame, (1, 2, 0))
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        if self.shape_type_input == 'chw':
            frame = np.transpose(frame, (2, 0, 1))

        return frame

class ScaledZeroToOneFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class NumpyToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(NumpyToPytorch, self).__init__(env)

    def observation(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

class WarpFramePyTorch(gym.ObservationWrapper):
    def __init__(self, env, gray_scale=False, image_size=64):
        """
        Warp frames to (image_size, image_size) as done in the Nature paper
        and later work.
        :param env: (Gym Environment) the environment
        :param image_size: (int) the size of the image
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = image_size
        self.height = image_size
        self.gray_scale = gray_scale
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(1 if gray_scale else 3, self.height, self.width),
            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        if self.gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return frame[:, :, None]
        else:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame = np.transpose(frame, (2, 0, 1))
            return frame

# https://github.com/fabiopardo/tonic/blob/48a7b72757f7629c8577eb062373854a3663c255/tonic/environments/wrappers.py
class TimeFeature(gym.Wrapper):
    '''Adds a notion of time in the observations.
    It can be used in terminal timeout settings to get Markovian MDPs.
    Only vector features representations though, doesn't work with pixel images
    '''

    def __init__(self, env, max_steps, low=-1, high=1):
        super().__init__(env)
        dtype = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, low).astype(dtype),
            high=np.append(self.observation_space.high, high).astype(dtype))
        self.max_episode_steps = max_steps
        self.steps = 0
        self.low = low
        self.high = high

    def reset(self, **kwargs):
        self.steps = 0
        observation = self.env.reset(**kwargs)
        observation = np.append(observation, self.low)
        return observation

    def step(self, action):
        assert self.steps < self.max_episode_steps
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        prop = self.steps / self.max_episode_steps
        v = self.low + (self.high - self.low) * prop
        observation = np.append(observation, v)
        return observation, reward, done, info