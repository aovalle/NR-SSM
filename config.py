from sys import platform
from griddly import gd
import pprint

# Griddly
# Rataban levels: 0:det & no death, 1:det, 2:stoch, 3:stoch & irrelevance, 4:det v2, 5: det v3, 6:stoch v2, 7:stoch & harder
# Four rooms levels:    0:classic, 1:room division with fire, 2:classic with random enemy, 3:fire with random enemy
#                       4:classic with a-star enemy, 5:fire with a-star enemy, 6:classic with both enemies, 7: fire with both enemies
GRIDDLY_ENVS = ['Rataban', 'Four-Rooms']
GRIDDLY_POMDP = [False, True]
GRIDDLY_SPEC = {'env':   GRIDDLY_ENVS[1],
                'level': 0,
                'pomdp': GRIDDLY_POMDP[0]}

SPECS = ['Griddly', 'Planet']

def get_params(args):
    config_name = SPECS[0]
    print(config_name)
    #print(args.config_name)

    try:
        #config = globals()[f'{args.config_name}Config']()
        config = globals()[f'{config_name}Config']()
    except:
        raise ValueError("`{}` is not a valid config ID".format(config_name))

    config.set_logdir(args.logdir)
    return config

class Config(object):
    def __init__(self):
        # Agents
        agent = ['av1', 'av2', 'av3', 'av4', 'q_rhe', 'q_rhe2', 'q_rhe2_grad'] #, 'rhe', 'dv2']
        self.agent = agent[6]

        # Bookkeeping
        self.wandb_log = False if platform == 'win32' else True
        self.id = 0                     # Experiment id
        self.seed = None #0                   # Random seed
        self.logdir = "log"
        self.wandb_interval_step = 10 #50       # Log in wandb every N steps
        self.wandb_interval_episode = 1     # Log in wandb every N episodes
        #self.log_interval_step = 50         # Log every N steps
        self.print_interval_step = 50        # Print losses every N steps
        self.print_interval_episode = 1#10     # Print losses every N episodes
        self.video_interval_episode = 1     # Save video every N episides
        self.save_interval_episode = 50     # Save models every N episodes
        # Other intervals
        self.train_interval_step = 5 #50 #4        # To train via backprop every N steps
        #self.target_update_step = 100#50                # update every BACKPROP steps (8000 grayscale, 24000 color)
        #self.polyak_tau = 1  # .0001#.005              # If tau=1 then it becomes a hard update
        self.eval_interval_episode = 10     # To eval/test every N episodes
        self.eval_episodes = 1              # Number of episodes to test the agent
        # RL
        self.polyak_tau = .0001  # If tau=1 then it becomes a hard update
        self.val_target_update_interval = 100  # if hard then update every N gradient steps

        # Environment
        self.env_name = None
        self.env_seed = None
        self.max_episode_len = 500
        self.rand_gen_level = False
        self.rand_gen_level_interval_episode = 25     # Generate random levels every N episodes
        self.rand_agent_interval_episode = None         # Randomize agent position every N episodes
        self.rand_goal_interval_episode = None          # Randomize goal position every N episodes
        self.gen_seed = None # 345

        # Architecture
        # RSSM
        self.rssm_type = 'categorical'      # 'categorical' or 'gaussian'
        #self.rssm_type = 'gaussian'      # 'categorical' or 'gaussian'
        self.det_size = 100                 # Deterministic RNN state
        # Stochastic VAE state
        self.category_size = 16
        self.class_size = 16
        self.stoch_size = 32 if self.rssm_type == 'gaussian' else self.category_size * self.class_size
        self.min_std = 0.1          # For gaussian ssm
        # Network specs
        # World model
        self.rssm_net = {
            'node_size':100,
            'activation':'ELU',
        }
        self.embed_net = {                  # Produces obs embedding
            'embed_size':100,
            'activation':'ELU',             # 'ELU'/'ReLU'
        }
        self.decoder = {
            'activation':'ELU',
            'out_dist': 'gaussian',
        }
        self.reward_net = {                 # Predicts rewards
            'layers':3,
            'node_size':100,
            'activation':'ELU',
            'out_dist': 'gaussian',
        }
        self.discount_net = {
            'layers':3,
            'node_size':100,
            'activation':'ELU',
            'out_dist': 'bernoulli',
        }
        self.q_net = {
            'layers': 3,
            'node_size': 100,
            'activation': 'ELU',
            'out_dist': 'gaussian',
        }

        # Training
        #self.n_train_episodes = 700
        #self.n_seed_episodes = 10
        self.training_type ='steps'                     # To train every N steps ('steps') or at the end of an episode ('episodic')
        self.n_seed_steps = 4000  # 20000            # To initialize experience replay
        self.n_steps = 500000#int(1e6)  # Number of training steps (excluding seed steps)
        self.n_episodes = 1000          # Alternatively number of training episodes (exclusing seed steps)
        self.n_train_epochs = 5
        self.lr_wm = 2e-4
        self.lr_value = 8e-5
        #self.epsilon = 1e-7  # 1e-4             # Adam epsilon
        self.grad_clip_norm = 100
        # KL
        self.kl_balance = True
        self.kl_balance_alpha = 0.8
        self.free_nats = None
        # Loss scaling
        self.global_kl_weight = 0  # Global KL weight (0 to disable)
        self.kl_weight = 1.0
        self.rew_weight = 1.0
        self.term_weight = 10.0                  # Discounting
        self.value_weight = 1.0

        # Replay buffer
        self.buffer_size = 10000 #10 ** 6
        self.batch_size = 22 if platform == 'win32' else 50         # Batch size samples for representation (planet 50)
        self.batch_seq_len = 8

        # Planning
        plan = [None, 'ac', 'rhe', 'ac+rhe']
        self.planning = plan[0]
        self.plan_horizon = 20
        # RHE
        self.generations = 1
        self.candidates = 300
        self.mutation_rate = 0.5
        self.shift_buffer = True
        if self.agent in ['dqn', 'q_rhe', 'dqn_seed_rhe', 'q_rhe2', 'q_rhe2_grad']:
            self.how_to_eval_rh = 'action'
        elif self.agent in ['av1', 'av2', 'av4', 'a2c']:
            self.how_to_eval_rh = 'no_action'
        # DQN Planner
        self.n_plans = 10
        self.dqn_plan_horizon = 20

        # RL
        self.gamma = 0.99

        # NCE
        #tdim: G->L'
        nce_methods = ['nce-glt', 'ince-ggt', 'ince-glt', 'ince-gl-ll']
        self.nce_method = nce_methods[1]
        if self.nce_method == 'ince-gl-ll':
            self.embed_net['embed_size'] = 128
        self.lr_nce = 3e-4
        self.embedding_nce = self.embed_net['embed_size'] #256


        # Misc
        self.time_awareness = False      # Whether the agent should consider time step as part of the state


    def set_logdir(self, logdir):
        self.logdir = logdir

    def __repr__(self):
        return pprint.pformat(vars(self))

class GriddlyConfig(Config):
    def __init__(self):
        super().__init__()

        self.environment = GRIDDLY_SPEC['env']
        self.env_level = GRIDDLY_SPEC['level']
        self.pomdp = GRIDDLY_SPEC['pomdp']
        self.env_name = f'GDY-{self.environment}-v0'

        # Representations
        g_repr = [gd.ObserverType.SPRITE_2D, gd.ObserverType.BLOCK_2D, gd.ObserverType.VECTOR]
        self.render_repr = g_repr[0]    # how it's rendered
        self.agent_repr = g_repr[0]     # what the agent sees

        # Wrappers
        self.unwrapped = False          # overrides everything
        self.scale_obs = False          # scale obs to 0-1
        self.to_planet_obs = False      # scale obs to -0.5-0.5 like in planet
        self.grayscale = False
        self.stack = False
        self.stacked_frames = 4
        self.max_episode_len = 3000 #500 #100
        if not self.pomdp:
            self.frame_size = (64, 64)
        else:
            self.frame_size = (28, 28)

        # Env reward scheme
        classic_rew = {'death':-1, 'step':0, 'goal':1}
        cai_rew = {'death':-100, 'step':-1, 'goal':0}
        self.g_rew_scheme = classic_rew

class PlanetConfig(GriddlyConfig):
    def __init__(self):
        super().__init__()

        # Architecture
        self.rssm_type = 'gaussian'
        self.stoch_size = 32
        self.det_size = 200
        # World model
        self.rssm_net = {
            'node_size': 200,
            'activation': 'ReLU',
        }
        self.embed_net = {  # Produces obs embedding
            'embed_size': 200,
            'activation': 'ReLU',  # 'ELU'/'ReLU'
        }
        self.decoder = {
            'activation': 'ReLU',
            'out_dist': 'gaussian',
        }
        self.reward_net = {  # Predicts rewards
            'layers': 2,
            'node_size': 200,
            'activation': 'ReLU',
            'out_dist': 'gaussian',
        }
        # Buffer
        self.batch_size = 50
        self.batch_seq_len = 20
        # Training
        self.training_type = 'episodic'
        self.n_train_epochs = 100
        self.lr_wm = 6e-4  # 1e-3
        self.epsilon = 1e-7
        self.grad_clip_norm = 100
        # KL
        self.kl_balance = False
        self.free_nats = 3  # Allowed deviation in KL divergence
        self.global_kl_weight = 0.1  # Global KL weight (0 to disable)
        # Loss scaling
        self.kl_weight = 1.0
        self.rew_weight = 1.0
        self.term_weight = 0