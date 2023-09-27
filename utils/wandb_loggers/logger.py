import os
import numpy as np

from collections import deque

class Logger:
    def __init__(self, args):
        self.args = args
        self.wandb = args.wandb_log

        self.hist_return = 0    # During training
        self.total_steps = 0    # During training

        self.running_return = 0
        self.running_mean_return = deque(maxlen=10)  # During training
        self.running_mean_outcome = deque(maxlen=10)  # During training
        self.running_mean_steps = deque(maxlen=10)  # During training
        self.best_train_return = -np.inf  # During training

        self.start_time = 0
        self.duration = 0
        self.running_reward = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.n_success = 0

    def update_stats(self, *args):

        episode, episode_return, episode_steps, episode_outcome = args

        self.hist_return += episode_return
        self.total_steps += episode_steps
        self.running_mean_outcome.append(episode_outcome)
        self.running_mean_steps.append(episode_steps)
        self.running_mean_return.append(episode_return)

        self.running_return = 0.99 * self.running_return + 0.01 * episode_return if episode > 0 else episode_return
        self.best_train_return = max(self.best_train_return, episode_return)
        self.max_episode_reward = max(self.max_episode_reward, episode_return)

        if self.running_reward == 0:
            self.running_reward = episode_return
        else:
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_return

        self.last_10_ep_rewards.append(int(episode_return))
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.last_10_ep_reward = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')
        else:
            self.last_10_ep_reward = 0

        self.n_success = self.n_success+1 if episode_outcome == 1 else self.n_success

    def log_episode(self, *args):
        learning_episodes, episode_return, episode_outcome, episode_steps = args

        self.wandb.log({'Episodic return': episode_return, 'Episode': learning_episodes})
        self.wandb.log({'Running mean steps': np.mean(self.running_mean_steps), 'Episode': learning_episodes})
        self.wandb.log({'Running mean outcome': np.mean(self.running_mean_outcome), 'Episode': learning_episodes})
        self.wandb.log({'Running mean return': np.mean(self.running_mean_return), 'Episode': learning_episodes})
        self.wandb.log({'Running return': self.running_return, 'Episode': learning_episodes})
        self.wandb.log({'Max return': self.best_train_return, 'Episode': learning_episodes})
        self.wandb.log({'Historical cumulative return': self.hist_return, 'Episode': learning_episodes})
        self.wandb.log({'Episode outcome': episode_outcome, 'Episode': learning_episodes})
        self.wandb.log({'Steps per episode': episode_steps, 'Episode': learning_episodes})
        self.wandb.log({'Total steps': self.total_steps, 'Episode': learning_episodes})

        self.wandb.log({'Ep running reward': self.running_reward, 'Episode': learning_episodes})
        self.wandb.log({'Max episode reward': self.max_episode_reward, 'Episode': learning_episodes})
        self.wandb.log({'MA reward last 10 episodes': self.last_10_ep_reward, 'Episode': learning_episodes})

        self.wandb.log({'Success rate': (self.n_success/(learning_episodes+1))*100, 'Episode': learning_episodes})