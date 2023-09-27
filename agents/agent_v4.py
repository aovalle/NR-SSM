import torch
import os
import numpy as np
from collections import defaultdict

from torch.distributions.kl import kl_divergence
from torch.nn.functional import one_hot
from torch.nn import functional as F
from torch.distributions import Normal, Independent

from buffers.full_trajectory_buffer_negatives import TrajectoryBuffer
from nn.encoder import ConvEmbedder
from nn.decoder import ObservationModel, GenericPredictorModel
from nn.classifiers import BilinearClassifier
from ssm.rssm_danijar import RSSM
from utils.common import logging
from utils.common.image_proc import visualization
from utils.common.image_proc.visualization import VideoRecorder
from utils.common.network_utils import get_channels, get_parameters, no_grad_in
from utils.wandb_loggers.logger import Logger
from planners.rolling_horizon import RollingHorizon
from utils.common.stats_utils import calculate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

class AgentV4(object):
    def __init__(self, env, test_env, args):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.action_space = env.action_space.n
        self.obs_space = env.observation_space.shape

        if args.seed is not None:
            self.env.seed(args.seed)
            self.test_env.seed(2 ** 31 - 1 - args.seed)

        self.step = 0
        self.backprop_steps = 0
        self.episodes = 0
        self.learning_episodes = 0

        # Architecture components WM
        self.channels = get_channels(self.obs_space, args)
        self.embedder = ConvEmbedder(self.channels, args.embed_net['embed_size'], args).to(device)
        self.decoder = ObservationModel(args.det_size + args.stoch_size, self.channels, args).to(device)
        self.rssm = RSSM(self.action_space, args).to(device)
        self.reward_model = GenericPredictorModel(args.det_size + args.stoch_size, 1, args.reward_net).to(device)
        self.discount_model = GenericPredictorModel(args.det_size + args.stoch_size, 1, args.discount_net).to(device)

        # NCE
        # self.bi_classifier = torch.nn.Linear(self.embedder.embedding_size, 128).to(device)
        # Input 1: latent, Input 2: local features
        local_dim = self.embedder.local_layer_depth
        self.bi_classifier = BilinearClassifier(args.det_size + args.stoch_size, self.embedder.local_layer_depth).to(device)
        self.fc_obsrew_embedding = torch.nn.Sequential(
            torch.nn.Linear(local_dim + 1, local_dim),
            torch.nn.ReLU(),
        ).to(device)

        # Optimizers
        self.rep_nets = [self.rssm, self.embedder, self.fc_obsrew_embedding]
        self.dec_nets = [self.decoder, self.reward_model, self.discount_model]
        #self.wm_nets = [self.rssm, self.embedder, self.reward_model, self.discount_model]
        self.rep_optim = torch.optim.Adam([{'params': get_parameters(self.rep_nets), 'lr': args.lr_wm},
                                       {'params': self.bi_classifier.parameters(), 'lr': args.lr_nce}])  # , eps=args.epsilon)
        # self.wm_optim = torch.optim.Adam([{'params': get_parameters(self.wm_nets), 'lr': args.lr_wm},
        #                                    {'params': self.bi_classifier.parameters(),
        #                                     'lr': args.lr_nce}])  # , eps=args.epsilon)
        # self.dec_optim = torch.optim.Adam(list(self.decoder.parameters()), lr=args.lr_wm, eps=1e-5)
        self.dec_optim = torch.optim.Adam(get_parameters(self.dec_nets), lr=args.lr_wm, eps=1e-5)

        # Architecture components Control
        self.rhe = RollingHorizon(args, self.action_space, self.rssm, self.reward_model)

        # Memories
        self.buffer = TrajectoryBuffer(self.channels, args.frame_size, args.buffer_size, args.batch_size, args.batch_seq_len)

        # Logs & stats
        self.wandb = args.wandb_log
        self.modeldir = os.path.join(self.args.logdir, 'model')
        self.best_eval_return = -np.inf
        self.video_recorder = VideoRecorder(args.logdir, with_recon=True)
        self.metrics = defaultdict(float) #{}
        self.logger = Logger(args)

        # Misc
        if self.args.free_nats:
            self.free_nats = torch.full((1,), args.free_nats, dtype=torch.float32).to(device)

    def decide_action(self, latent):
        action, _ = self.rhe.plan(latent)
        onehot_action = one_hot(torch.LongTensor([action]).to(device), num_classes=self.action_space).float()
        return action, onehot_action

    def train(self):
        training = False
        step = 0
        while self.step <= self.args.n_seed_steps + self.args.n_steps and self.learning_episodes < self.args.n_episodes:
            ###############################
            # Set-up episode
            # == Stats
            episode_return = 0
            episode_steps = 0
            # == Init env
            if self.args.rand_gen_level:
                lv = generate_level(self.args.env_level, 'A')  # Generate level randomizing agent's location
                obs = self.env.reset(level_string=lv)
            else:
                obs = self.env.reset()
            done = False
            self.video_recorder.flush()

            # == Start the buffer
            # o_t, a_t-1=0, r_t, d_t, i_t (starting state)
            initial = True
            self.buffer.append(obs, 0, 0, done, initial)

            # == Start episode
            # Init h_t-1, s_t-1, a_t-1
            prev_h = torch.zeros(1, self.args.det_size, requires_grad=False).to(device)
            prev_s = torch.zeros(1, self.args.stoch_size, requires_grad=False).to(device)
            prev_a = torch.zeros(1, self.action_space, requires_grad=False).to(device)
            while not done:
                if training:
                    with torch.no_grad():
                        # Generate latent to take an action
                        embed_obs = self.embedder(obs)
                        # o_t, h_t-1, s_t-1, a_t-1 -> z_t (h_t, s_t)
                        latent, prev_h, prev_s = self.rssm.obs_to_latent(embed_obs, prev_h, prev_s, prev_a, not initial)
                        action, onehot_action = self.decide_action(latent)
                        prev_a = onehot_action
                        # Visualization
                        self.video_recorder.append(obs, self.decoder(latent).mean.detach())
                else:
                    action = self.env.action_space.sample()

                next_obs, reward, done, info = self.env.step(action)
                initial = False
                self.buffer.append(next_obs, action, reward, done, initial) # o_t+1, a_t, r_t+1, d_t+1, i_t+1

                # Learn perception and control every N steps (1st condition) or at the end of the episode (2nd condition)
                #if self.step % self.args.train_interval_step == 0 and training:
                if ((self.step % self.args.train_interval_step == 0 and self.args.training_type == 'steps') or \
                        (done and self.args.training_type == 'episodic')) and training:
                    self.backprop_steps += 1
                    self.learn()
                if not training and self.step >= self.args.n_seed_steps:     # Stop collection mode
                    done = True
                    training = True
                    step = -1

                obs = next_obs

                step += 1
                episode_steps += 1
                episode_return += reward
                self.step += 1  # Steps from the beginning of times

            # To do once training has started
            if self.backprop_steps > 0:
                tt = info['TerminationType']
                episode_outcome = 0 if tt == 'timeout' else -1 if tt == 'fail' else 1

                self.logger.update_stats(self.learning_episodes, episode_return, episode_steps, episode_outcome)

                # Wandb - episode stats
                if self.learning_episodes % self.args.wandb_interval_episode == 0 and self.wandb:
                    self.metrics['episode'] = self.learning_episodes
                    self.wandb.log(self.metrics)
                    self.logger.log_episode(self.learning_episodes, episode_return, episode_outcome, episode_steps)
                # Evaluate
                # if self.learning_episodes % self.args.eval_interval_episode == 0:
                #     self.evaluate()
                # Save models #todo
                # if self.learning_episodes % self.args.save_interval_episode == 0:
                #     self.save_state_dict(os.path.join(self.modeldir, 'final'))
                # Save video
                if self.learning_episodes % self.args.video_interval_episode == 0:
                    # Obtain latent from last seen observation
                    embed_obs = self.embedder(obs)
                    latent = self.rssm.obs_to_latent(embed_obs, prev_h, prev_s, prev_a, True)[0]
                    self.video_recorder.append(obs, self.decoder(latent).mean.detach())
                    self.video_recorder.record(self.learning_episodes)
                    self.video_recorder.flush()
                # Log print
                if self.learning_episodes % self.args.print_interval_episode == 0:
                    print(f'Total steps {self.step} Episode {self.learning_episodes} Steps {episode_steps} Return {episode_return}')

                self.learning_episodes += 1
            self.episodes += 1

    def learn(self):
        self.metrics = defaultdict(float)   # init metrics
        for epoch in range(self.args.n_train_epochs):
            # Draw sequence chunks originally {(o_t:T, a_t-1:T-1, r_t:T, notdone_t:T} ~ D
            # for NCE {(o_t:T+1, a_t-1:T, r_t:T+1, notdone_t:T+1} ~ D
            obs, neg_obs, actions, rewards, nonterminals, noninit, neg_rewards = self.buffer.sample()
            onehot_actions = one_hot(actions.squeeze(-1).long(), num_classes=self.action_space).float()
            # Learn World Model
            self.learn_perception(obs, neg_obs, onehot_actions, rewards, nonterminals, noninit, neg_rewards)
            #print(f'Episode {self.learning_episodes} Step {self.steps} Epoch {epoch}')
        self.metrics = {k: v/self.args.n_train_epochs for k, v in self.metrics.items()} # avg by num of epochs


    ##################################################
    ### W o r l d   M o d e l ########################
    ##################################################

    def learn_perception(self, obs, neg_obs, actions, rewards, nonterminals, noninit, neg_rewards):
        # o_t, a_t - 1 = 0, r_t, d_t, i_t(starting state)
        # Encode obs o_t:T+1
        embobs = self.embedder(obs, fmaps=True)
        f_t = embobs['out'][:-1]  # (seq, batch, embedding size)  (t:T)
        f_t_next = embobs['f5'][1:]  # (seq, batch, 6,6,128)         (t+1:T+1)
        f_neg = self.embedder(neg_obs[:-1], fmaps=True)['f5']  # (seq, batch, 6,6,128) (len |T|)

        # o_t:T, a_t-1:T-1, d_t:T, i_t:T -> Obtain h_t:T, s_t:T
        prior_trans, post_trans = self.rssm.get_transitions(f_t, actions[:-1], noninit[:-1])
        # -> (seq, batch, s + h dim)
        latent = torch.cat((post_trans['h'], post_trans['s']), dim=-1)

        # Get predictive distributions p(.|z_t:T)
        #obs_dist = self.decoder(latent)             # p(o_t|s_t,h_t) ≈ p(o_t|s_t, h_t-1, s_t-1, a_t-1) filtering
        # reward_dist = self.reward_model(latent)     # p(r_t|s_t,h_t) (either r_t or r_t+1 determined by the reward targets below)
        # nonterm_dist = self.discount_model(latent)

        # Losses
        nce_loss = self._nce_loss(latent, f_t_next, rewards[1:], f_neg, neg_rewards[:-1])
        #reconstruction = self._obs_loss(obs_dist, obs)                 # o_t:T
        # reward_loss = self._reward_loss(reward_dist, rewards[:-1])          # r_t:T
        # nonterm_loss = self._pcont_loss(nonterm_dist, nonterminals[:-1])    # d_t:T

        # kl_loss = self._kl_loss(prior_trans['dist_params'], post_trans['dist_params'])
        # # Global KL (Planet)
        # if self.args.global_kl_weight != 0:
        #     kl_loss += self._global_kl_loss(post_trans['dist_params'])

        loss = nce_loss #+ reward_loss + nonterm_loss #+ kl_loss

        # Backprop
        # self._backprop(self.wm_optim, loss, self.wm_nets + [self.bi_classifier])
        self._backprop(self.rep_optim, loss, self.rep_nets + [self.bi_classifier])

        # DECODER To check what it's capturing
        # with no_grad_in(self.wm_nets, self.bi_classifier):
        with no_grad_in(self.rep_nets, self.bi_classifier):
            obs_dist = self.decoder(latent.detach())
            reconstruction = self._obs_loss(obs_dist, obs[:-1].detach())
            reward_dist = self.reward_model(latent.detach())  # p(r_t|s_t,h_t) (either r_t or r_t+1 determined by the reward targets below)
            nonterm_dist = self.discount_model(latent.detach())
            reward_loss = self._reward_loss(reward_dist, rewards[:-1])  # r_t:T
            nonterm_loss = self._pcont_loss(nonterm_dist, nonterminals[:-1])  # d_t:T
            loss_dec = reconstruction + reward_loss + nonterm_loss #+ kl_loss
            # rec_obs = self.decoder(latent.detach()).mean
            # target_obs = obs / 255. - 0.5
            # reconstruction = F.mse_loss(rec_obs, target_obs[1:-1].detach(), reduction='sum')
            self.dec_optim.zero_grad()
            loss_dec.backward()
            self.dec_optim.step()
            # self.metrics['decoder_loss'] += reconstruction.detach().item()

        self.metrics['loss'] += loss.item()

    def _nce_loss(self, f_t, f_t_next, r, f_neg, r_neg):
        # Reshape to batcherize sequences (seq,batch,...) -> (seq*batch,...)
        f_t = f_t.reshape(-1, f_t.shape[-1] # (seq,batch,latent dim) -> (seq*batch,latent dim)
                          ).unsqueeze(1).unsqueeze(1).expand(f_t.size(0)*f_t.size(1),   # -> (s*b, 1,1,lat dim)
                                                             f_t_next.size(2),
                                                             f_t_next.size(3), -1)      # -> (s*b, 6,6, lat dim)

        r_t_next = r.reshape(-1, r.shape[-1]).unsqueeze(1).unsqueeze(1).expand(r.size(0)*r.size(1),
                                                             f_t_next.size(2), f_t_next.size(3), -1)
        r_neg = r_neg.reshape(-1, r_neg.shape[-1]).unsqueeze(1).unsqueeze(1).expand(r_neg.size(0) * r_neg.size(1),
                                                                               f_t_next.size(2), f_t_next.size(3), -1)

        f_t_next = f_t_next.reshape(-1, *f_t_next.shape[-3:])
        f_neg = f_neg.reshape(-1, *f_neg.shape[-3:])

        f_r_pos = torch.cat([f_t_next, r_t_next], dim=-1)
        f_pos = self.fc_obsrew_embedding(f_r_pos)
        f_r_neg = torch.cat([f_neg, r_neg], dim=-1)
        f_neg = self.fc_obsrew_embedding(f_r_neg)

        # Create the target (batch*seq*2, 6,6)
        target = torch.cat((torch.ones_like(f_t[:, :, :, 0]), torch.zeros_like(f_t[:, :, :, 0])), dim=0).to(device)
        # Prepare classifier inputs
        x1 = torch.cat([f_t, f_t], dim=0)  # (b*s*2, 6,6, lat size)
        x2 = torch.cat([f_pos, f_neg], dim=0)  # (b*s*2, 6,6,128)
        shuffled_idxs = torch.randperm(len(target))  # b*s*2
        x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]

        # Compute scores f(x,y) (b*s*2, 6,6)
        scores = self.bi_classifier(x1, x2)
        nce_loss = torch.nn.BCEWithLogitsLoss()(scores, target)
        self.metrics['nce_loss'] += nce_loss.detach().item()
        # Accuracy
        preds = torch.sigmoid(scores.detach())
        self.metrics['nce_accuracy'] += calculate_accuracy(preds, target).item()
        return nce_loss

    ''' log p(o|s) '''
    def _obs_loss(self, dist, target_obs):
        target_obs = target_obs / 255. - 0.5
        # (seq, batch, c, w, h) -> (seq, batch)
        reconstruction = -torch.mean(dist.log_prob(target_obs))
        self.metrics['recon_loss'] += reconstruction.item()
        return reconstruction

    def _reward_loss(self, dist, target_rewards):
        # (seq, batch, c, w, h) -> (seq, batch)
        reward_loss = -torch.mean(dist.log_prob(target_rewards))
        self.metrics['reward_loss'] += reward_loss.item()
        return self.args.rew_weight * reward_loss

    def _pcont_loss(self, dist, target_nonterms):
        nonterm_loss = -torch.mean(dist.log_prob(target_nonterms.float()))
        self.metrics['pcont_loss'] += nonterm_loss.item()
        return self.args.term_weight * nonterm_loss

    def _kl_loss(self, prior_param, posterior_param):
        prior_dist = self.rssm.build_dist(prior_param)
        posterior_dist = self.rssm.build_dist(posterior_param)
        kl = kl_divergence(posterior_dist, prior_dist).mean().detach()  # mean over (batch seq and batch size)
        if self.args.kl_balance:
            alpha = self.args.kl_balance_alpha
            # To train prior towards the representations
            post_dist_nograd = self.rssm.build_dist(posterior_param.detach())
            kl_prior = kl_divergence(post_dist_nograd, prior_dist).mean()
            # To regularize the representations towards the priors
            prior_dist_nograd = self.rssm.build_dist(prior_param.detach())
            kl_post = kl_divergence(posterior_dist, prior_dist_nograd).mean()

            if self.args.free_nats:
                kl_prior = torch.max(kl_prior, kl_prior.new_full(kl_prior.size(), self.args.free_nats))
                kl_post = torch.max(kl_post, kl_post.new_full(kl_post.size(), self.args.free_nats))

            kl_loss = (alpha * kl_prior) + ((1 - alpha) * kl_post)
        else:
            # ∑_t=1 DKL[q(s_t|o<=t, a<=t) || p(s_t|s_t-1, a_t-1)]
            # TODO: ASSIGN TO KL_LOSS =, IF NO GRADIENTS ARE AFFECTED
            _kl = kl_divergence(posterior_dist, prior_dist).mean()
            if self.args.free_nats:
                kl_loss = torch.max(_kl, _kl.new_full(_kl.size(), self.args.free_nats))
            else:
                kl_loss = _kl
        # todo: log entropy of dists
        self.metrics['kl'] += kl.item()
        self.metrics['kl_loss'] += kl_loss.item()
        return self.args.kl_weight * kl_loss

    def _global_kl_loss(self, posterior_param):
        '''
        # Global prior (Used in Planet)
        # fixed global prior to prevent the posteriors from collapsing in near deterministic envs
        # alleviates overfitting to the initially small training dataset and grounds the state
        # beliefs (since posteriors and temporal priors are both learned, they could drift in
        # latent space). Another interpretation of this is to define the prior at each time step
        # as a product of the learned temporal prior and the global fixed prior

        ∑_t=1 DKL[q(s_t|o<=t, a<=t) || p(s)]
        '''
        posterior_dist = self.rssm.build_dist(posterior_param)
        batch_seq_size = posterior_dist.batch_shape[0]  # (batch seq, batch size)
        # N(0,I)
        global_prior = Independent(Normal(torch.zeros(batch_seq_size, self.args.batch_size, self.args.stoch_size).to(device),
                              torch.ones(batch_seq_size, self.args.batch_size, self.args.stoch_size).to(device)), 1)
        global_kl = kl_divergence(posterior_dist, global_prior).mean()
        self.metrics['global_kl_loss'] += global_kl.item()
        return self.args.global_kl_weight * global_kl

    def _backprop(self, optim, loss, networks=None):
        optim.zero_grad()
        loss.backward()
        if self.args.grad_clip_norm and networks is not None:
            for net in networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.grad_clip_norm)
        optim.step()