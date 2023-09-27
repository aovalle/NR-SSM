import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import copy
from torch.nn.functional import one_hot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RollingHorizon():

    def __init__(self, args, action_space, rssm, eval_model):
        self.args = args
        self.horizon = args.plan_horizon
        self.generations = args.generations
        self.candidates = args.candidates
        self.mutation_rate = args.mutation_rate
        self.shift_buffer = args.shift_buffer

        self.rssm = rssm
        self.eval_model = eval_model

        self.action_space = action_space
        self.curr_rollout = None

        self.trial_returns = []

    def plan(self, latent, seed_plan=None):
        # Clone to have one per candidate (1, latent) -> (candidates, latent)
        latent = latent.expand(self.candidates, -1)

        # If we're performing RHE based on an initial sequence provided by the Actor critic
        if seed_plan is not None:
            self.curr_rollout = seed_plan
        # But for vanilla RHE, generate an initial action sequence if there's no current seq or if shift buffer is off
        elif self.curr_rollout is None or not self.shift_buffer:
            self.curr_rollout = np.random.randint(self.action_space, size=self.horizon)
        else:
            self.curr_rollout = self.apply_shift_buffer(np.copy(self.curr_rollout))

        # Maximize
        overall_largest_return = float("-inf")
        # Number of generations to eval before selecting a sequence
        for _ in range(self.generations):
            # Mutate rollout (candidates, horizon)
            rollouts = self.mutate(self.curr_rollout)
            # From (candidates, horizon) to torch (horizon, candidates, action dim) '''
            rollouts = torch.FloatTensor(rollouts.transpose()).unsqueeze(-1).to(device)
            # Evaluate rollouts and get best one
            best_rollout, best_return = self.evaluate_sequences(latent, rollouts)

            if best_return > overall_largest_return:
                overall_largest_return = best_return
                self.curr_rollout = best_rollout.cpu().numpy().flatten().astype(int)

        return self.curr_rollout[0], self.curr_rollout

    def evaluate_sequences(self, latent, action_seqs, sample_seq=False):
        # Simulate trajectory
        # h_t+1:T+1 = f(h_t, s_t, a_t)      s_t+1:T+1 ~ p(s_t+1:T|h_t+1:T)
        sim_trajectory = self.rssm.simulate(action_seqs, latent)
        latent = torch.cat((sim_trajectory['h'], sim_trajectory['s']), dim=-1)  # -> (horizon, candidates, h+s)
        # p(r_t|s_t) (horizon, candidates, 1)
        if self.args.how_to_eval_rh in ['action']: # e.g. r = Q(z,a)
            returns = self.eval_model(latent, action_seqs).mean.sum(dim=0)  # -> (candidates, 1)
        elif self.args.how_to_eval_rh in ['onehot_action']:   # e.g. r = reward(z,oh(a))
            onehot_action = one_hot(action_seqs.squeeze(-1).long(), num_classes=self.action_space).float()
            returns = self.eval_model(latent, onehot_action).mean.sum(dim=0)  # -> (candidates, 1)
        elif self.args.how_to_eval_rh in ['no_action']:   # e.g. r = reward(z)
            returns = self.eval_model(latent).mean.sum(dim=0)           # -> (candidates, 1)
        else:
            raise NotImplementedError

        # Select the one that maximizes the return
        best_return, idx_best = returns.max(dim=0)
        best_action_seq = action_seqs[:, idx_best, :]
        return best_action_seq, best_return

    def mutate(self, rollout):
        # Clone sequence
        rollouts = np.tile(rollout, (self.candidates-1, 1))
        # Generate indices to mutate
        idx = np.random.rand(*rollouts.shape) < self.mutation_rate
        # Generate new actions and place them accordingly
        rollouts[idx] = np.random.randint(self.action_space, size=len(idx[idx == True]))
        # Add original sequence to mutated ones
        rollouts = np.row_stack((rollout, rollouts))
        return rollouts

    def apply_shift_buffer(self, rollout):
        # append new random action at the end
        sf_rollout = np.append(rollout, np.random.randint(self.action_space))
        # remove first action
        sf_rollout = np.delete(sf_rollout, 0)
        return sf_rollout
