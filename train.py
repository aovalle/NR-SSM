import numpy as np
import torch
import argparse
import wandb
import time

import config
from env.envs import GriddlyGymEnv
from agents.agent import Agent
from agents.agent_v2 import AgentV2
from agents.agent_v3 import AgentV3
from agents.agent_v4 import AgentV4
from agents.agent_q import AgentDQN
from agents.agent_q_v2 import AgentDQN2
from agents.agent_q_v2_grads import AgentDQN2_GRAD
from utils.common import logging

#wandb_log = True

# ======= Init config =======
#SPECS = ['Griddly', 'Planet']
#config_spec = SPECS[0]

time_stamp = time.strftime("%d%m%Y-%H%M")
parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default='log-{}'.format(time_stamp))
#parser.add_argument("--config_name", type=str, default=config_spec)
args = parser.parse_args()
args = config.get_params(args)

# ====== Monitoring =======
if args.wandb_log:
    args.wandb_log = wandb.init(project='ThesisComparisons', dir='{}/monitoring'.format(args.logdir), config=args, name='NCE-RH-{}'.format(time_stamp))

# ======= Seeds =======
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

# ======= Logs ========
print(time.ctime())
print(args)
print(args.logdir)
logging.init_dirs(args.logdir)

# ======= Environment =========
env = GriddlyGymEnv(args)
test_env = GriddlyGymEnv(args, registered=True)
action_space = env.action_space.n
obs_dims = env.observation_space.shape
print('action space ', action_space, ' obs space ', obs_dims)

# ======= Agent =======
if args.agent == 'av1':
    print('agent v1')
    agent = Agent(env, test_env, args)
elif args.agent == 'av2':
    print('agent v2')
    agent = AgentV2(env, test_env, args)
elif args.agent == 'av3':
    print('agent v3')
    agent = AgentV3(env, test_env, args)
elif args.agent == 'av4':
    print('agent v4')
    agent = AgentV4(env, test_env, args)
elif args.agent == 'q_rhe':
    print('agent q_rhe')
    agent = AgentDQN(env, test_env, args)
elif args.agent == 'q_rhe2':
    print('agent q_rhe v2')
    agent = AgentDQN2(env, test_env, args)
elif args.agent == 'q_rhe2_grad':
    print('agent q_rhe v2')
    agent = AgentDQN2_GRAD(env, test_env, args)

agent.train()