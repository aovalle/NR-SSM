import gym
import gym.spaces
import numpy as np

'''
This manually terminates the environment (done=True)
if it encounters certain user defined rewards
'''
class GriddlyControlReset(object):
    def __init__(self, env, rewards):
        self.env = env
        self.rewards = rewards

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward in self.rewards:
            done = True
            # check that the reward isn't coming due to timeout
            if 'TerminationType' not in info:
                info['TerminationType'] = 'fail' if reward == self.rewards[0] else 'success'
        return obs, reward, done, info

class GriddlyChangeCost(object):
    def __init__(self, env, original, new, optimality=False):
        self.env = env
        self.original = original
        self.new = new
        self.optimality = optimality

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Set new reward value
        if reward == self.original:
            reward = self.new
        # If optimality is on set positive reward as cost 0
        if reward == 1 and self.optimality:
            reward = 0
        return obs, reward, done, info

'''
Changes the rewards defined in a particular griddly env to other rewards
It can receive a list of values to change
'''
class GriddlyChangeRewards(object):
    def __init__(self, env, original, new):
        assert len(original) == len(new)
        self.env = env
        self.original = np.array(original)
        self.new = np.array(new)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Set new reward value
        if reward in self.original:
            reward = self.new[self.original==reward][0]
        return obs, reward, done, info


'''
LEGACY (this has been fixed in updated versions)
Does what GriddlyControlReset does but the reward is extracted from
the info dictionary to bypass Griddly's reward bug
'''
class GriddlyRewardBug(gym.Wrapper):
    def __init__(self, env, rewards):
        super(GriddlyRewardBug, self).__init__(env)
        self.rewards = rewards

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # we have to check if reward is actually in the dictionary
        if reward == 0 and len(info['History']) > 0:
            # add the rewards (in most cases it'll be 0 because reward is rare)
            reward = np.sum([x["Reward"] for x in info['History']]).item()
        # if the reward is one of those we expect to terminate the episode then do so
        if reward in self.rewards:
            done = True
        return obs, reward, done, info

