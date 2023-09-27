import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot

from ssm.rssm_nce import RSSM_NCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RSSM_DQN(RSSM_NCE):
    def __init__(self, action_space, args):
        super().__init__(action_space, args)

    def get_dqn_plan(self, init_latent, value_model, n_plans):
        # Clone to have one per plan candidate (1, h+s) -> (n_plans, h+s)
        init_latent = init_latent.expand(n_plans, -1)
        # (candidates, h+s) -> (candidates, h) (candidates, s)
        h, s = torch.split(init_latent, [self.args.det_size, self.args.stoch_size], dim=-1)

        # Obtain N action sequence(s) in simulated space
        qvalue_seqs, action_seqs = self._simulate_with_dqn(value_model, h, s, stoch=True if n_plans > 1 else False)

        returns = qvalue_seqs.sum(dim=0)  # -> (candidates, 1)
        # Select the one that maximizes the return
        best_return, idx_best = returns.max(dim=0)
        best_action_seq = action_seqs[:, idx_best, :].flatten()
        # Return full action sequence and first action
        return best_action_seq, best_action_seq[0].item()

    # Starts at a real latent state and simulates decision making via the q net
    def _simulate_with_dqn(self, value_model, h, s, stoch):
        actions = []
        qvalues = []
        for t in range(self.args.dqn_plan_horizon):
            latent = torch.cat((h, s), dim=-1).detach()

            q = value_model(latent).mean    # (n plans, action space)
            if not stoch:
                qvalue, action = torch.max(q, dim=1)
                onehot_action = one_hot(action, num_classes=self.action_space).float()
                qvalue, action = qvalue.unsqueeze(-1), action.unsqueeze(-1)
            else:
                # Boltzmann exploration
                probs = F.softmax(q, dim=1)
                action = torch.multinomial(probs, 1)    # (n plans, 1)
                qvalue = torch.gather(q, 1, action)     # Grab corresponding qvalue -> (n plans, 1)
                onehot_action = one_hot(action.squeeze(-1), num_classes=self.action_space).float()
            # Obtain h_t = f(h_t-1, s_t-1, a_t-1), compute p(s_t|h_t) and sample an s_t
            prior = self.get_prior(h, s, onehot_action, nonterminal=True)

            qvalues.append(qvalue)
            actions.append(action)
            s, h = prior['s'], prior['h']

        qvalues = torch.stack(qvalues, dim=0)   # -> (horizon, n plans, 1)
        actions = torch.stack(actions, dim=0)   # -> (horizon, n plans, 1)
        return qvalues, actions
