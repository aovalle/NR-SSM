import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from utils.common.network_utils import Flatten, conv_output_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvEmbedder(nn.Module):
    def __init__(self, channels, embedding_size, args):
        super().__init__()

        activation = getattr(nn, args.embed_net['activation'])
        self.input_shape = (channels, *args.frame_size)
        self.embedding_size = embedding_size

        if args.pomdp:  # 28x28
            self.conv = nn.Sequential(
                        nn.Conv2d(channels, 32, 4, stride=2),
                        activation(),
                        nn.Conv2d(32, 64, 4, stride=2),
                        activation(),
                        nn.Conv2d(64, 128, 4, stride=2),
                        activation(),
                        nn.Conv2d(128, 256, 1, stride=2),
                        activation(),
                    )
        else:   # 64x64
            net = [nn.Conv2d(channels, 32, 4, stride=2),
                   nn.ReLU(),
                   nn.Conv2d(32, 64, 4, stride=2),
                   nn.ReLU(),
                   nn.Conv2d(64, 128, 4, stride=2),
                   nn.ReLU(),
                   nn.Conv2d(128, 256, 4, stride=2),
                   nn.ReLU(),
                   Flatten()]

        # Get number of elements
        conv_output_numel, _ = conv_output_size(nn.Sequential(*net), self.input_shape)
        net.append(nn.Linear(conv_output_numel, self.embedding_size))
        self.net = nn.Sequential(*net)

    @property
    def local_layer_depth(self):
        return self.net[4].out_channels

    def forward(self, obs, fmaps=False):
        obs = obs / 255. - 0.5
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        # (seq, batch, c, w, h)
        #print('obs ', obs.shape)
        x = obs.reshape(-1, *self.input_shape)                      # -> (seq*batch, c, w, h)
        if not fmaps:
            x = self.net(x)                                             # -> (seq*batch, obs embed)
            return x.reshape((*obs.shape[:-3], self.embedding_size))    # -> (seq, batch, obs embed)
        else:
            #print('x ', x.shape)
            f5 = self.net[:6](x)            # -> (seq*batch, 128,6,6)
            #print('f5 ', f5.shape)
            f7 = self.net[6:8](f5)          # -> (seq*batch, 256,2,2)
            #print('f7 ', f7.shape)
            out = self.net[8:](f7)          # -> (seq*batch, embedding)
            #print('out ', out.shape)
            #print(obs.shape, *obs.shape[:-3])
            #print(f5.shape, f5.reshape((*obs.shape[:-3], *f5.shape[-3:])).shape)
            return {
                'f5': f5.reshape((*obs.shape[:-3], *f5.shape[-3:])).permute(0, 1, 3, 4, 2),    # -> (seq, batch, 6,6,128)
                'f7': f7.reshape((*obs.shape[:-3], *f7.shape[-3:])).permute(0, 1, 3, 4, 2),    # -> (seq, batch, 2,2,256)
                'out': out.reshape((*obs.shape[:-3], self.embedding_size))    # -> (seq, batch, obs embed)
            }
