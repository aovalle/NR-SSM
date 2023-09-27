import gym
import os
import torch
from griddly import GymWrapperFactory, gd
from utils.common.env.wrappers import env_wrappers, obs_wrappers, griddly_wrappers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def GriddlyGymEnv(args, registered=False):
    dirname, filename = os.path.split(os.path.abspath(__file__))

    if args.environment == 'Rataban':
        if not args.pomdp:
            env_path = os.path.join(dirname, "griddly", "rataban", "rataban.yaml")
        elif args.pomdp:
            env_path = os.path.join(dirname, "griddly", "rataban", "rataban_pomdp.yaml")
    elif args.environment == 'Four-Rooms':
        env_path = os.path.join(dirname, "griddly", "tasks", "four_rooms.yaml")

    if not registered:
        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml(
            args.environment,
            env_path,
            level=args.env_level,
            global_observer_type=args.render_repr, #gd.ObserverType.SPRITE_2D,
            player_observer_type=args.agent_repr, #gd.ObserverType.VECTOR,
        )

    env = gym.make(args.env_name)
    env.enable_history(True)
    # Random level generation
    env.reset()
    # if args.rand_gen_level:
    #     env.reset(level_string=generate_level(args.env_level, 'A'))
    # else:
    #     env.reset()

    if not args.unwrapped:
        # Set time limit and termination reward
        if args.time_awareness:
            env = env_wrappers.LimitDuration(env, args.max_episode_len, termination_reward=args.g_rew_scheme['death'])
        else:
            env = env_wrappers.LimitDuration(env, args.max_episode_len)
        # Rescale and Grayscale
        env = obs_wrappers.RescaleFrame(env, frame_size=args.frame_size, grayscale=args.grayscale)

        if args.to_planet_obs:
            env = obs_wrappers.PlanetScaledFrame(env, ret_np_or_torch='torch')
        elif args.scale_obs:
            env = obs_wrappers.ScaledZeroToOneFrame(env)
            env = obs_wrappers.NumpyToPytorch(env)
        # if args.stacked_frames: #3,64,64
        #     env = obs_wrappers.FrameStack(env, args.stacked_frames)

        # Rewards
        death_cost = args.g_rew_scheme['death'] #-100
        step_cost = args.g_rew_scheme['step'] #-1
        goal_cost = args.g_rew_scheme['goal'] #0
        # Change 0 rewards to -1 cost (e.g. p(O=1|s,a) = exp(-1))
        # And control as inference change +1 rewards to cost 0 thus p(O=1|s,a)=1
        # So currently rewards are like this:
        # r = 0 -> p(O=1|s,a) = exp(r) = 1
        # r = -1 -> p(O=1|s,a) = exp(r) = .36...
        # r = -500 -> p(O=1|s,a) = exp(r) = 7.12...e-218
        #env = griddly_wrappers.GriddlyChangeRewards(env, original=[-1, 0, 1], new=[death_cost, step_cost, goal_cost])
        # Reset env if observing specific rewards (currently: death (-1) and success (1))
        env = griddly_wrappers.GriddlyControlReset(env, rewards=[death_cost, goal_cost])


    if args.env_seed is not None:
        env.seed(args.env_seed)

    return env
