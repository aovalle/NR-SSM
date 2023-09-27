import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryBuffer():
    def __init__(self, channels, frame_size, size, batch_size, seq_len):
        self.size = size
        self.obs_shape = (channels, *frame_size)
        self.batch_size = batch_size
        self.seq_len = seq_len + 1

        self.obs = np.empty((size, *self.obs_shape), dtype=np.float32) #NOTE: np.uint8 for 50 x 50 batch (lazy frames)
        self.actions = np.empty((size, 1), dtype=np.float32)#dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size,), dtype=bool)
        self.noninit = np.empty((size,), dtype=bool)
        #self.non_terminals = np.empty((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False   # if memory full

    # Store as numpy
    def append(self, obs, action, reward, done, initial):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.noninit[self.idx] = not initial
        self.idx = (self.idx + 1) % self.size       # where to append next
        self.full = self.full or self.idx == 0      # whether it's full

    # Return as pytorch
    def sample(self):
        # Get a batch with indices for each sequence
        batch_seq_idxs = np.asarray([self._get_seq_idxs(self.seq_len) for _ in range(self.batch_size)])
        # Get batch
        batch = self._get_batch(batch_seq_idxs, self.batch_size, self.seq_len)
        return [torch.as_tensor(item).to(device) for item in batch]

    # Sample uniformly to return the indices of a valid sequence chunk
    def _get_seq_idxs(self, seq_len):
        valid_idx = False
        while not valid_idx:
            valid_max_idx = self.size if self.full else self.idx - seq_len
            init_idx = np.random.randint(0, valid_max_idx)
            seq_idxs = np.arange(init_idx, init_idx + seq_len) % self.size
            # Check data does not cross the buffer index. We don't want data from non-contiguous
            # in time sequences
            # idx indicates where we'll write next so in case it pointed to the first seq_idx
            # that's ok since it'll be overwritten only until next append()
            valid_idx = not self.idx in seq_idxs[1:]
        return seq_idxs

    def _get_batch(self, seq_idxs, batch_size, seq_len):
        # unroll indices from shape(seq len, batch size) to get obs
        seq_idxs = seq_idxs.transpose().reshape(-1)

        # obs = torch.as_tensor(self.obs[seq_idxs].astype(np.float32))
        # # Undo discretization for visual observations
        # obs = img.preprocess_obs(obs, self.bit_depth)
        # obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])  # (seq, batch,

        obs = self.obs[seq_idxs].reshape(seq_len, batch_size, *self.obs_shape) #/ 255. - 0.5
        actions = self.actions[seq_idxs].reshape(seq_len, batch_size, 1)
        rewards = self.rewards[seq_idxs].reshape(seq_len, batch_size, 1)
        nonterminals = self.nonterminals[seq_idxs].reshape(seq_len, batch_size, 1)
        noninit = self.noninit[seq_idxs].reshape(seq_len, batch_size, 1)

        return obs, actions, rewards, nonterminals, noninit