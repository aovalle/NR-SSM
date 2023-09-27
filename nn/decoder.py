import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, Bernoulli

class ObservationModel(nn.Module):
    def __init__(self, latent_size, channels, args):
        super().__init__()

        activation = getattr(nn, args.decoder['activation'])
        embedding_size = args.embed_net['embed_size']
        self.embedding_size = embedding_size
        self.frame_shape = (channels, *args.frame_size)

        self.fc = nn.Linear(latent_size, embedding_size)

        if args.frame_size == (28,28): #2,3,4,4->26 | 2,3,5,4 -> 28
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(embedding_size, 128, 2, stride=2),
                activation(),
                nn.ConvTranspose2d(128, 64, 3, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, 5, stride=2),
                activation(),
                nn.ConvTranspose2d(32, channels, 4, stride=2),
            )
        elif args.frame_size == (64,64):
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                activation(),
                nn.ConvTranspose2d(128, 64, 5, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, 6, stride=2),
                activation(),
                nn.ConvTranspose2d(32, channels, 6, stride=2),
            )
        elif args.frame_size == (84,84):
            raise NotImplementedError
            # self.conv = nn.Sequential(
            #     nn.ConvTranspose2d(embedding_size, 128, 6, stride=2),
            #     activation(),
            #     nn.ConvTranspose2d(128, 64, 7, stride=2),
            #     activation(),
            #     nn.ConvTranspose2d(64, 32, 7, stride=2),
            #     activation(),
            #     nn.ConvTranspose2d(32, channels, 8, stride=2),
            # )


    def forward(self, latent):
        # (seq, batch, s+h)
        out = self.fc(latent)                                           # -> (seq, batch, obs embed)
        out = out.view(-1, self.embedding_size, 1, 1)                   # -> (seq*batch, obs embed, 1, 1)
        mean = self.conv(out)                                           # -> (seq*batch, channels, w, h)
        mean = mean.reshape((*latent.shape[:-1], *self.frame_shape))    # -> (seq, batch, c, w, h)
        # Construct multivariate gaussian
        # batch shape = (seq, batch), event shape = (c, w, h)
        dist = Independent(Normal(mean, 1), len(self.frame_shape))

        return dist

class GenericPredictorModel(nn.Module):

    def __init__(self, input_dim, output_space, arch):
        super().__init__()
        self.out_dist = arch['out_dist']
        activation = getattr(nn, arch['activation'])

        fc = [nn.Linear(input_dim, arch['node_size'])]
        fc += [activation()]
        for i in range(1, arch['layers']):
            fc += [nn.Linear(arch['node_size'], arch['node_size'])]
            fc += [activation()]
        fc += [nn.Linear(arch['node_size'], output_space)]
        self.fc = nn.Sequential(*fc)

    def forward(self, latent, action=None):
        #out = self.fc(torch.cat([hidden, state], dim=1)).squeeze(dim=1)
        if action is not None:
            latent = torch.cat([latent, action], dim=-1)
        dist_param = self.fc(latent)
        if self.out_dist == 'gaussian':
            # batch shape = (seq len, batch), event shape = (output space)
            return Independent(Normal(dist_param, 1), 1)
        if self.out_dist == 'bernoulli':
            return Independent(Bernoulli(logits=dist_param), 1)
        if self.out_dist is None:
            return dist_param
        raise NotImplementedError

class QvalueModel(nn.Module):

    def __init__(self, input_dim, output_space, arch):
        super().__init__()
        self.out_dist = arch['out_dist']
        activation = getattr(nn, arch['activation'])

        fc = [nn.Linear(input_dim, arch['node_size'])]
        fc += [activation()]
        for i in range(1, arch['layers']):
            fc += [nn.Linear(arch['node_size'], arch['node_size'])]
            fc += [activation()]
        fc += [nn.Linear(arch['node_size'], output_space)]
        self.fc = nn.Sequential(*fc)

    def forward(self, latent, action=None):
        # (seq len, batch size, action space)
        dist_param = self.fc(latent)
        # Q(z,.) -> Q(z,a) (seq len, batch size, 1)
        if action is not None:
            dist_param = dist_param.gather(2, action.to(torch.long))
        # batch shape = (seq len, batch), event shape = (1 if gathered action space otherwise)
        if self.out_dist == 'gaussian':
            return Independent(Normal(dist_param, 1), 1)
        if self.out_dist is None:
            return dist_param