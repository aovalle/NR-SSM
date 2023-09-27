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
from nn.decoder import ObservationModel, GenericPredictorModel, QvalueModel
from nn.classifiers import BilinearClassifier
from ssm.rssm_dqn import RSSM_DQN
from utils.common import logging
from utils.common.image_proc import visualization
from utils.common.image_proc.visualization import VideoRecorder
from utils.common.network_utils import get_channels, get_parameters, no_grad_in
from utils.wandb_loggers.logger import Logger
from planners.rolling_horizon import RollingHorizon
from utils.common.stats_utils import calculate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

class AgentDQN(object):
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
        self.real_backprop_steps = 0
        self.episodes = 0
        self.learning_episodes = 0

        # Architecture components WM
        self.channels = get_channels(self.obs_space, args)
        self.embedder = ConvEmbedder(self.channels, args.embed_net['embed_size'], args).to(device)
        self.decoder = ObservationModel(args.det_size + args.stoch_size, self.channels, args).to(device)
        self.rssm = RSSM_DQN(self.action_space, args).to(device)
        self.reward_model = GenericPredictorModel(args.det_size + args.stoch_size, 1, args.reward_net).to(device)
        self.discount_model = GenericPredictorModel(args.det_size + args.stoch_size, 1, args.discount_net).to(device)
        # Architecture components Control
        self.value_model = QvalueModel(args.det_size + args.stoch_size, self.action_space, args.q_net).to(device)
        self.target_value_model = QvalueModel(args.det_size + args.stoch_size, self.action_space, args.q_net).to(device)

        # NCE
        self.nce_nets = []
        # self.bi_classifier = torch.nn.Linear(self.embedder.embedding_size, 128).to(device)
        if args.nce_method == 'ince-ggt':
            self.classifier = torch.nn.Linear(args.det_size + args.stoch_size, args.embedding_nce).to(device)
        if args.nce_method in ['ince-glt', 'ince-gl-ll']:
            self.classifier = torch.nn.Linear(args.det_size + args.stoch_size, self.embedder.local_layer_depth).to(device)
            if args.nce_method == 'ince-gl-ll':
                self.classifier2 = torch.nn.Linear(args.det_size + args.stoch_size, self.embedder.local_layer_depth).to(device)
                self.nce_nets.append(self.classifier2)
        elif args.nce_method == 'nce-glt':
            # Input 1: latent, Input 2: local features
            self.classifier = BilinearClassifier(args.det_size + args.stoch_size, self.embedder.local_layer_depth).to(device)

        # Optimizers
        self.nce_nets.append(self.classifier)
        self.rep_nets = [self.rssm, self.embedder]
        self.dec_nets = [self.decoder, self.reward_model, self.discount_model]
        #self.wm_nets = [self.rssm, self.embedder, self.decoder, self.reward_model, self.discount_model]
        #self.wm_optim = torch.optim.Adam(get_parameters(self.wm_nets), lr=args.lr_wm)
        #self.value_optim = torch.optim.Adam(self.value_model.parameters(), lr=args.lr_value)

        # self.optim = torch.optim.Adam([{'params': get_parameters(self.wm_nets), 'lr': args.lr_wm},
        #                                {'params': self.value_model.parameters(), 'lr': args.lr_value}])#, eps=args.epsilon)
        self.rep_optim = torch.optim.Adam([{'params': get_parameters(self.rep_nets), 'lr': args.lr_wm},
                                           {'params': get_parameters(self.nce_nets), 'lr': args.lr_nce}])
        self.dec_optim = torch.optim.Adam([{'params': get_parameters(self.dec_nets), 'lr': args.lr_wm},
                                           {'params': self.value_model.parameters(), 'lr': args.lr_value}])


        # Architecture components Control
        self.rhe = RollingHorizon(args, self.action_space, self.rssm, self.value_model)

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
        if self.args.agent == 'q_rhe':
            action, _ = self.rhe.plan(latent)
            onehot_action = one_hot(torch.LongTensor([action]).to(device), num_classes=self.action_space).float()
        elif self.args.agent == 'dqn':  # DQN is trained with sequences but does not plan so it's effectively model-free
            # TODO: to try argmax, boltzmann expl, and samples
            qvalues = self.value_model(latent).mean
            #qvalues = self.value_model(latent).sample()

            # Argmax selection
            action = torch.argmax(qvalues).item()
            onehot_action = one_hot(torch.LongTensor([action]).to(device), num_classes=self.action_space).float()

            # Boltzmann exploration
            probs = F.softmax(qvalues, dim=1)
            action = torch.multinomial(probs, 1).item()
            onehot_action = one_hot(torch.LongTensor([action]).to(device), num_classes=self.action_space).float()
        elif self.args.agent == 'dqn_plan': # DQN plans in latent space using the SSM
            _, action = self.rssm.get_dqn_plan(latent, self.value_model, self.args.n_plans)
            onehot_action = one_hot(torch.LongTensor([action]).to(device), num_classes=self.action_space).float()
        elif self.args.agent == 'dqn_seed_rhe': # Generate initial plan with DQN
            dqn_plan, _ = self.rssm.get_dqn_plan(latent, self.value_model, 1)
            # Use it to initialize RHE
            action, _ = self.rhe.plan(latent, dqn_plan.cpu().numpy())
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
            self.real_backprop_steps += 1
            # Draw sequence chunks {(o_t:T, a_t-1:T-1, r_t:T, notdone_t:T} ~ D
            obs, neg_obs, actions, rewards, nonterminals, noninit, _ = self.buffer.sample()
            onehot_actions = one_hot(actions.squeeze(-1).long(), num_classes=self.action_space).float()
            # Learn World Model
            self.learn_perception(obs, neg_obs, actions, onehot_actions, rewards, nonterminals, noninit)
            # wm_loss, latents = self.learn_perception(obs, neg_obs, onehot_actions, rewards, nonterminals, noninit)
            # value_loss = self.learn_control(latents, actions, rewards, nonterminals)
            # # Backprop
            # self._backprop(self.optim, wm_loss + value_loss, self.wm_nets + [self.value_model])
            # Update value target network
            self._update_target_value_net()
            #print(f'Episode {self.learning_episodes} Step {self.steps} Epoch {epoch}')
        self.metrics = {k: v/self.args.n_train_epochs for k, v in self.metrics.items()} # avg by num of epochs


    ##################################################
    ### W o r l d   M o d e l ########################
    ##################################################

    def learn_control(self, latents, actions, rewards, nonterminals):
        # Get predicted q-value distribution prediction     p(q_t|z_t,a_t)
        # Q(z_t:T-1, .) but gather a -> Q(z_t:T-1, a_t:T-1)
        value_dist = self.value_model(latents[:-1], actions[1:])    # batch shape (seq len,batch size) event shape (1)
        #decoded_value = self.value_model(latents[:-1]).mean
        #decoded_value = decoded_value.gather(2, actions[1:-1].to(torch.long)).squeeze(-1)

        # Get target
        with torch.no_grad():
            # Q(z_t+1:T, .)     p(q_t+1|z_t+1)
            # get mean for each action
            next_value_target = self.target_value_model(latents[1:]).mean
            # argmax_a Q(z',a')
            next_value_target_max = next_value_target.detach().max(2, keepdim=True)[0]

            # Q(z_t:T-1, a_t:T-1) = r_t+1:T + nt_t+1:T γ Q(z_t+1:T, a_t+1:T)
            target_value = rewards[1:] + (nonterminals[1:] * self.args.gamma * next_value_target_max.detach())

        value_loss = self._value_loss(value_dist, target_value.detach())
        return value_loss

    def _value_loss(self, dist, target_values):
        value_loss = -torch.mean(dist.log_prob(target_values))
        self.metrics['value_loss'] += value_loss.detach().item()
        return self.args.value_weight * value_loss

    def learn_perception(self, obs, neg_obs, actions, onehot_actions, rewards, nonterminals, noninit):
        # o_t, a_t - 1 = 0, r_t, d_t, i_t(starting state)
        # for NCE {(o_t:T+1, a_t-1:T, r_t:T+1, notdone_t:T+1} ~ D
        # Encode obs o_t:T+1
        embobs = self.embedder(obs, fmaps=True)
        fg_t = embobs['out'][:-1]  # (seq, batch, embedding size)  (t:T)
        fl_t = embobs['f5'][:-1]  # (seq, batch, embedding size)  (t:T)
        fg_t_next = embobs['out'][1:]  # (seq, batch, embedding size)  (t+1:T+1)
        fl_t_next = embobs['f5'][1:]  # (seq, batch, 6,6,128)         (t+1:T+1)
        fl_neg = self.embedder(neg_obs[:-1], fmaps=True)['f5']  # (seq, batch, 6,6,128) (len |T|)

        # o_t:T, a_t-1:T-1, d_t:T, i_t:T -> Obtain h_t:T, s_t:T
        prior_trans, post_trans = self.rssm.get_transitions(fg_t, onehot_actions[:-1], noninit[:-1])
        # -> (seq, batch, s + h dim)
        latent = torch.cat((post_trans['h'], post_trans['s']), dim=-1)

        # Get predictive distributions p(.|z_t:T)
        #obs_dist = self.decoder(latents)             # p(o_t|s_t,h_t) ≈ p(o_t|s_t, h_t-1, s_t-1, a_t-1) filtering
        #reward_dist = self.reward_model(latents)     # p(r_t|s_t,h_t)
        #nonterm_dist = self.discount_model(latents)

        # Losses
        if self.args.nce_method == 'nce-glt':
            nce_loss = self._nce_loss(latent, fl_t_next, fl_neg)
        elif self.args.nce_method == 'ince-ggt':
            nce_loss = self._infonce_loss(latent, fg_t_next)
        elif self.args.nce_method == 'ince-glt':
            nce_loss = self._infonce_loss(latent, fl_t_next)
        elif self.args.nce_method == 'ince-gl-ll':
            _, post_trans_l = self.rssm.get_transitions_flocal(fl_t, onehot_actions[:-1], noninit[:-1])
            latent_l = torch.cat((post_trans_l['h'], post_trans_l['s']), dim=-1)
            latent_l = latent_l.reshape(self.args.batch_seq_len, self.args.batch_size, fl_t.size(2), fl_t.size(3), -1)
            nce_loss = self._infonce_loss(latent, fl_t_next, latent_l)

        # Losses
        # reconstruction = self._obs_loss(obs_dist, obs)                 # o_t:T
        # reward_loss = self._reward_loss(reward_dist, rewards)          # r_t:T
        # nonterm_loss = self._pcont_loss(nonterm_dist, nonterminals)    # d_t:T
        #
        # kl_loss = self._kl_loss(prior_trans['dist_params'], post_trans['dist_params'])
        # # Global KL (Planet)
        # if self.args.global_kl_weight != 0:
        #     kl_loss += self._global_kl_loss(post_trans['dist_params'])
        #
        # loss = reconstruction + reward_loss + nonterm_loss + kl_loss
        loss = nce_loss

        # Backprop
        self._backprop(self.rep_optim, loss, self.rep_nets + self.nce_nets)
        #self._backprop(self.wm_optim, loss, self.wm_nets)

        self.metrics['loss'] += loss.detach().item()

        with no_grad_in(self.rep_nets, self.nce_nets):
            obs_dist = self.decoder(latent.detach())
            reconstruction = self._obs_loss(obs_dist, obs[:-1].detach())
            reward_dist = self.reward_model(latent.detach())  # p(r_t|s_t,h_t) (either r_t or r_t+1 determined by the reward targets below)
            nonterm_dist = self.discount_model(latent.detach())
            reward_loss = self._reward_loss(reward_dist, rewards[:-1])  # r_t:T
            nonterm_loss = self._pcont_loss(nonterm_dist, nonterminals[:-1])  # d_t:T
            loss_dec = reconstruction + reward_loss + nonterm_loss  # + kl_loss
            # NOTE: DETACHED
            value_loss = self.learn_control(latent.detach(), actions[:-1], rewards[:-1], nonterminals[:-1])
            self._backprop(self.dec_optim, loss_dec + value_loss, self.dec_nets + [self.value_model])
            #self._backprop(self.optim, wm_loss + value_loss, self.wm_nets + [self.value_model])

        #return loss, latent


    def _nce_loss(self, f_t, f_t_next, f_neg):
        # Reshape to batcherize sequences (seq,batch,...) -> (seq*batch,...)
        f_t = f_t.reshape(-1, f_t.shape[-1] # (seq,batch,latent dim) -> (seq*batch,latent dim)
                          ).unsqueeze(1).unsqueeze(1).expand(f_t.size(0)*f_t.size(1),   # -> (s*b, 1,1,lat dim)
                                                             f_t_next.size(2),
                                                             f_t_next.size(3), -1)      # -> (s*b, 6,6, lat dim)
        f_t_next = f_t_next.reshape(-1, *f_t_next.shape[-3:])
        f_neg = f_neg.reshape(-1, *f_neg.shape[-3:])

        # Create the target (batch*seq*2, 6,6)
        target = torch.cat((torch.ones_like(f_t[:, :, :, 0]), torch.zeros_like(f_t[:, :, :, 0])), dim=0).to(device)
        # Prepare classifier inputs
        x1 = torch.cat([f_t, f_t], dim=0)  # (b*s*2, 6,6, lat size)
        x2 = torch.cat([f_t_next, f_neg], dim=0)  # (b*s*2, 6,6,128)
        shuffled_idxs = torch.randperm(len(target))  # b*s*2
        x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]

        # Compute scores f(x,y) (b*s*2, 6,6)
        scores = self.classifier(x1, x2)
        nce_loss = torch.nn.BCEWithLogitsLoss()(scores, target)
        self.metrics['nce_loss'] += nce_loss.detach().item()
        # Accuracy
        preds = torch.sigmoid(scores.detach())
        self.metrics['nce_accuracy'] += calculate_accuracy(preds, target).item()
        return nce_loss

    def _infonce_loss(self, f_tg, f_t_next, f_tl=None):
        f_tg = f_tg.reshape(-1, f_tg.shape[-1])
        N = f_tg.size(0) # batch size
        # N x N
        if self.args.nce_method == 'ince-ggt':                      # f_tg, f_tg_next
            f_t_next = f_t_next.reshape(-1, f_t_next.shape[-1])
            scores = torch.matmul(self.classifier(f_tg), f_t_next.t())
            infonce_loss = F.cross_entropy(scores, torch.arange(N).to(device))
        elif self.args.nce_method in ['ince-glt', 'ince-gl-ll']:    # f_tg, ft_l_next
            # loop over features
            infonce_loss = 0
            f_t_next = f_t_next.reshape(-1, *f_t_next.shape[-3:])
            fx, fy = f_t_next.size(2), f_t_next.size(1)
            for y in range(fy):
                for x in range(fx):
                    scores = torch.matmul(self.classifier(f_tg), f_t_next[:, y, x, :].t())
                    step_loss = F.cross_entropy(scores, torch.arange(N).to(device))
                    infonce_loss += step_loss
            infonce_loss = infonce_loss / (fx * fy)
            if self.args.nce_method == 'ince-gl-ll':                # f_tl, ft_l_next
                f_tl = f_tl.reshape(-1, *f_tl.shape[-3:])
                step_loss = 0
                for y in range(fy):
                    for x in range(fx):
                        scores = torch.matmul(self.classifier2(f_tl[:, y, x, :]), f_t_next[:, y, x, :].t())
                        step_loss += F.cross_entropy(scores, torch.arange(N).to(device))
                infonce_loss += (step_loss/(fx*fy))
        self.metrics['nce_loss'] += infonce_loss.detach().item()

        return infonce_loss

    ''' log p(o|s) '''
    def _obs_loss(self, dist, target_obs):
        target_obs = target_obs / 255. - 0.5
        # (seq, batch, c, w, h) -> (seq, batch)
        reconstruction = -torch.mean(dist.log_prob(target_obs))
        self.metrics['recon_loss'] += reconstruction.detach().item()
        return reconstruction

    def _reward_loss(self, dist, target_rewards):
        # (seq, batch, c, w, h) -> (seq, batch)
        reward_loss = -torch.mean(dist.log_prob(target_rewards))
        self.metrics['reward_loss'] += reward_loss.detach().item()
        return self.args.rew_weight * reward_loss

    def _pcont_loss(self, dist, target_nonterms):
        nonterm_loss = -torch.mean(dist.log_prob(target_nonterms.float()))
        self.metrics['pcont_loss'] += nonterm_loss.detach().item()
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
        self.metrics['kl'] += kl.detach().item()
        self.metrics['kl_loss'] += kl_loss.detach().item()
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

    def _update_target_value_net(self):
        # Soft update
        if self.args.polyak_tau < 1:
            # Polyak averaging target value network
            self._polyak()
        # Hard udpate
        elif self.args.polyak_tau == 1 and self.real_backprop_steps % self.args.val_target_update_interval == 0:
            print(self.real_backprop_steps)  # TODO: to test
            raise
            # Substitute the target parameters for those of the trainable network
            self._polyak()

    ''' 
        θ_target = (1 - τ)*θ_target + τ*θ_train
        if τ=1 then it's equiv to hard update
        '''
    def _polyak(self):
        for target_param, train_param in zip(self.target_value_model.parameters(), self.value_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.args.polyak_tau) + train_param.data * self.args.polyak_tau)