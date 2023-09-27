import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, OneHotCategorical, Normal
from torch.nn.functional import one_hot

from utils.common.pytorch import OneHotCategoricalStraightThrough

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RSSM(nn.Module):
    def __init__(self, action_space, args):
        super().__init__()
        activation = getattr(nn, args.rssm_net['activation'])
        self.args = args
        self.action_space = action_space

        stoch_size = args.stoch_size * 2 if self.args.rssm_type == 'gaussian' else args.stoch_size  # If gaussian consider mu and std

        # Deterministic component ϕ(s+a) -> h
        # h' = f(h, s, a)
        # state+action -> embedding(s+a)
        self.fc_embed_stoch_state_action = nn.Sequential(
            nn.Linear(args.stoch_size + action_space, args.det_size),
            activation(),
        )
        # embedding(s+a) -> h
        self.rnn = nn.GRUCell(args.det_size, args.det_size)

        # Stochastic component | Prior ϕ(h) -> s_μ, s_σ
        # Compute p(s|h) get s_μ, s_σ
        self.fc_prior = nn.Sequential(
            nn.Linear(args.det_size, args.rssm_net['node_size']),      # h -> embedding(h)
            activation(),
            nn.Linear(args.rssm_net['node_size'], stoch_size),    # embedding(h) -> s_logits / s_μ, s_σ
        )

        # Stochastic component | Posterior ϕ(h, ϕ(o)) -> s_μ, s_σ
        # Compute q(s|h,o) get s_μ, s_σ
        self.fc_posterior = nn.Sequential(
            nn.Linear(args.det_size + args.embed_net['embed_size'], args.rssm_net['node_size']), # h+embedding(o) -> embedding(h+emb(o))
            activation(),
            nn.Linear(args.rssm_net['node_size'], stoch_size),                # embedding(h+emb(o)) -> s_logits / s_μ, s_σ
        )

    def obs_to_latent(self, embed_obs, prev_h, prev_s, prev_a, nonterminal):
        '''
        Takes an embedding and transforms it into its latent z (h,s). Used during decision making
        :param embed_obs:
        :param prev_h:
        :param prev_s:
        :param prev_a:
        :param nonterminal:
        :return:
        '''
        prior = self.get_prior(prev_h, prev_s, prev_a, nonterminal)    # Obtain h_t = f(h_t-1, s_t-1, a_t-1)
        post = self.get_posterior(embed_obs, prior['h'])               # Sample s_t ~ q(s_t|h_t, o_t)
        latent = torch.cat((post['h'], post['s']), dim=1)
        return latent, post['h'], post['s']

    def sample_latent_state(self, dist_params):
        '''
        Samples s ~ p(s_t|h_t) by receiving the parameter(s) of the p(s_t|h_t) distribution
        :param dist_params: logits (categorical) or concatenated mu and std (gaussian)
        :return: sample s
        '''

        if self.args.rssm_type == 'categorical':
            # (batch, stoch) -> (batch, category, class)
            logits = dist_params.reshape((-1, self.args.category_size, self.args.class_size))

            # dist = OneHotCategorical(logits=logits)
            # s = dist.sample()
            # # For straight through gradients
            # s += dist.probs - dist.probs.detach()
            # #s = torch.flatten(s, start_dim=-2, end_dim=-1)
            # # (batch, category, class) -> (batch, stoch)
            # # print(s)
            # # print(s.shape)

            # Equivalent using custom straight through class
            dist = OneHotCategoricalStraightThrough(logits=logits)
            s = dist.rsample()
            # print(s)
            # print(s.shape)

            s = s.reshape(-1, self.args.stoch_size)
            return s, dist_params
        elif self.args.rssm_type == 'gaussian':
            mean, std = torch.chunk(dist_params, 2, dim=-1)
            std = F.softplus(std) + self.args.min_std
            dist_params = torch.cat((mean, std), dim=-1)
            return mean + std * torch.randn_like(mean), dist_params


    '''
    h_t = f(h_t-1, s_t-1, a_t-1) -> p(s_t|h_t) -> s ~ p(s_t|h_t)
    '''
    def get_prior(self, prev_h, prev_s, prev_a, nonterminal):
        """ 1. compute deterministic hidden state """
        # get state-action embedding
        prev_sa = torch.cat([prev_s * nonterminal, prev_a], dim=1)
        prev_sa_embedding = self.fc_embed_stoch_state_action(prev_sa)
        # h_t = f(h_t-1, s_t-1, a_t-1)      (batch, h size)
        h = self.rnn(prev_sa_embedding, prev_h * nonterminal)
        """ 2. compute latent state prior (discrete categorical / continuous s_μ, s_σ) """
        # p(s_t|h_t) ≈ p(s_t|s_t-1, a_t-1, h_t-1)
        # Get categorical logits            (batch, s size = categories*classes)
        # OR mu and std for Gaussian
        dist_params = self.fc_prior(h)
        """ 3. sample from state prior (discrete s ~ Cat(logits) / continuous s ~ N(s_μ, s_σ)) """
        # Obtain distribution and sample s ~ p(s_t|h_t)     (batch, s size = categories*classes)
        s, dist_params = self.sample_latent_state(dist_params)

        #return {'logits':logits, 's':s, 'h':h}
        return {'dist_params':dist_params, 's':s, 'h':h}

    def get_posterior(self, obs, h):
        """ 1. embed hidden with obs ϕ(h_t, ϕ(o_t)) """
        hobs = torch.cat([h, obs], dim=1)
        """ 2. calculate latent state posterior """
        # q(s_t|h_t, o_t) ≈ p(s_t|o_t, s_t-1, a_t-1, h_t-1)
        dist_params = self.fc_posterior(hobs)
        """ 3. sample from state prior (continuous s ~ N(s_μ, s_σ)) """
        s, dist_params = self.sample_latent_state(dist_params)

        return {'dist_params': dist_params, 's': s, 'h': h}

    '''
        Used for learning transitions. Receives a batch of observations, actions and nonterm
        and obtains prior and posterior latent trajectories.
    '''
    def get_transitions(self, emb_obs, actions, nonterminals):
        # Init h_t-1, s_t-1
        prev_h = torch.zeros(self.args.batch_size, self.args.det_size).to(device)
        prev_s = torch.zeros(self.args.batch_size, self.args.stoch_size).to(device)
        prior_trajectory = []
        posterior_trajectory = []
        seq_len = actions.shape[0]  # (seq len, batch, onehot dim action)
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

    def simulate(self, action_seqs, init_latent):
        '''
        Open loop planning in latent space.
        Gathers the latent states obtained by unrolling a given action sequence
        :param action_seqs: (horizon, candidates, action dim=1)
        :param init_latent: (candidates, h+s)
        :return:
        '''
        horizon, candidates, _ = action_seqs.shape
        # (candidates, h+s) -> (candidates, h) (candidates, s)
        h, s = torch.split(init_latent, [self.args.det_size, self.args.stoch_size], dim=-1)
        # To one-hot (horizon*candidates, action space)
        action = one_hot(action_seqs.flatten().long(), num_classes=self.action_space).float()
        action = action.reshape(horizon, candidates, -1)
        prior_trajectory = []
        for t in range(horizon):
            prior = self.get_prior(h, s, action[t], nonterminal=True)
            prior_trajectory.append(prior)
            h, s = prior['h'], prior['s']
        prior_trajectory = self.latent_stack(prior_trajectory)
        return prior_trajectory

    def build_dist(self, dist_param):
        if self.args.rssm_type == 'categorical':
            # (batch seq, batch size, class*category) -> (batch seq*size, class, category) 7,50,256 -> 350,16,16
            logits = dist_param.reshape(-1, self.args.category_size, self.args.class_size)
            # batch shape (seq*batch), event shape (categories, classes)
            return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        elif self.args.rssm_type == 'gaussian':
            mean, std = torch.chunk(dist_param, 2, dim=-1)
            #
            #dist = Independent(Normal(mean, std), 1)
            #print(dist.event_shape, dist.batch_shape)
            #s = dist.sample()
            #print(s.shape)
            # event_shape (dim) batch_shape (batch seq, batch size) sample shape (batch seq, batch size, dim)
            return Independent(Normal(mean, std), 1)

    # list of dicts with tensors to dict of tensors
    def latent_stack(self, latents):
        return {
            'dist_params': torch.stack([l['dist_params'] for l in latents], dim=0),
            's': torch.stack([l['s'] for l in latents], dim=0),
            'h': torch.stack([l['h'] for l in latents], dim=0),
        }