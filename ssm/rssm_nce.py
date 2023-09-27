import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, OneHotCategorical, Normal
from torch.nn.functional import one_hot

from ssm.rssm_danijar import RSSM

from utils.common.pytorch import OneHotCategoricalStraightThrough

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RSSM_NCE(RSSM):
    def __init__(self, action_space, args):
        super().__init__(action_space, args)

    def get_transitions_flocal(self, emb_obs, actions, nonterminals):
        # Init h_t-1, s_t-1
        seq_len, batch_size, fx, fy, emb_dim = tuple(emb_obs.size())
        # (seq, batch, action dim) -> (seq, batch*fx*fy, action dim)
        actions = actions.unsqueeze(2).unsqueeze(2).expand(seq_len, batch_size, fx, fy, -1).reshape(seq_len, batch_size*fx*fy, -1)
        nonterminals = nonterminals.unsqueeze(2).unsqueeze(2).expand(seq_len, batch_size, fx, fy, -1).reshape(seq_len, batch_size*fx*fy, -1)
        emb_obs = emb_obs.reshape(seq_len, batch_size*fx*fy, -1)
        prev_h = torch.zeros(self.args.batch_size*fx*fy, self.args.det_size).to(device)
        prev_s = torch.zeros(self.args.batch_size*fx*fy, self.args.stoch_size).to(device)
        prior_trajectory = []
        posterior_trajectory = []
        for t in range(seq_len):
            prev_a = actions[t] * nonterminals[t]   # a_t-1, d_t
            # Obtain h_t = f(h_t-1, s_t-1, a_t-1), compute p(s_t|h_t) and sample an s_t
            prior = self.get_prior(prev_h, prev_s, prev_a, nonterminals[t])
            # Obtain p(s_t|o_t,h_t) and sample an s_t
            posterior = self.get_posterior(emb_obs[t], prior['h'])  # o_t, h_t
            prior_trajectory.append(prior)
            posterior_trajectory.append(posterior)
            prev_h, prev_s = posterior['h'], posterior['s']

        # (batch seq len, batch size, dim stoch/det state)
        # Format (stack them) from list to
        prior_trajectory = self.latent_stack(prior_trajectory)
        posterior_trajectory = self.latent_stack(posterior_trajectory)

        return prior_trajectory, posterior_trajectory