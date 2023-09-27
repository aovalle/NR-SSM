"""
    End episodes after specified number of steps.
    and optionally define a timeout reward
  """
class LimitDuration(object):

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
      # Inform about why it terminated
      info['TerminationType'] = 'timeout'
      # Set a termination reward if necessary
      if self._term_rew:
          reward = self._term_rew
    return observ, reward, done, info

  def reset(self, **kwargs):
    self._step = 0
    return self._env.reset(**kwargs)