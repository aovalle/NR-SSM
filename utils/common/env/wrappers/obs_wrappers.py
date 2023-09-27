import numpy as np
import torch
import gym
import gym.spaces
import cv2
from collections import deque

cv2.ocl.setUseOpenCL(False)

"""
    Rescale frames to 84x84 as done in the Nature DQN paper and later work.
    (By default chw -> pytorch)
"""
class RescaleFrame(gym.ObservationWrapper):
    def __init__(self, env, frame_size=(84, 84), grayscale=False, shape_type_input='chw'):
        super().__init__(env)
        self._width, self._height = frame_size
        self._grayscale = grayscale
        self.shape_type_input = shape_type_input

        n_channels = 1 if self._grayscale else 3

        if shape_type_input == 'chw':   # pytorch
            new_space = gym.spaces.Box(low=0, high=255, shape=(n_channels, self._height, self._width), dtype=np.uint8,)
        elif shape_type_input == 'hwc': # tf
            new_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, n_channels), dtype=np.uint8,)

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

'''
    Scales the observation to be within 0-1
'''
class ScaledZeroToOneFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

'''
    Converts numpy array to pytorch tensor
'''
class NumpyToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(NumpyToPytorch, self).__init__(env)

    def observation(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

''' 
    Transforms obs from 0-255 to -0.5-0.5 scale following Planet's code
    returns a torch tensor or a numpy vector 
'''
class PlanetScaledFrame(gym.ObservationWrapper):
    def __init__(self, env, ret_np_or_torch='torch'):
        super(PlanetScaledFrame, self).__init__(env)
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

'''
    Frame Stacking
    Modified from 
    https://raw.githubusercontent.com/openai/gym/master/gym/wrappers/frame_stack.py
    to work with griddly's wrapper and to have different final dimensionality
'''
class FrameStack(object):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        import gym
        env = gym.make('PongNoFrameskip-v0')
        env = FrameStack(env, 4)
        env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    def __init__(self, env, num_stack, lz4_compress=False):
        #super(FrameStack, self).__init__(env)
        self.env = env
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        # Modified stacking dimensions wrt the original version
        #low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        low = np.repeat(self.env.observation_space.low, num_stack, axis=0)
        #high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.env.observation_space.high, num_stack, axis=0)
        self.env.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.env.observation_space.dtype)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()

class LazyFrames(object):
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame