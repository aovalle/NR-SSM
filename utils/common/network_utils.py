'''
    17/aug/2021 version
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

def xavier_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def kaiming_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def kaiming_normal(m, nonlin="relu"):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

''' 
    Just a function to get the size of the output of the conv net part so it can be specified as input to the FC part
'''
def conv_output_size(net, input_shape):
    out = net(torch.zeros(1, *input_shape)).detach()
    return out.numel(), out.shape[1:]

class Flatten(nn.Module):
    def forward(self, x):
        #x = x.contiguous()
        #return x.view(x.size(0), -1)
        return x.reshape((x.size(0), -1))

class GaussianNN(nn.Module):
    """ Diagonal gaussian distribution parametrized by DNN. """

    def __init__(self, input_dim, output_dim, hidden_dim, act_fn="ReLU"):
        super(GaussianNN, self).__init__()

        self.act_fn = getattr(nn, act_fn)

        output_dim = 2*output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            self.act_fn(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            self.act_fn(),
            nn.Linear(hidden_dim[1], output_dim),
        )
        self.net.apply(xavier_uniform)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class IsotropicGaussian(nn.Module):
    """ Diagonal gaussian distribution with zero means. """

    def __init__(self, output_dim, std=1.0):
        super(IsotropicGaussian, self).__init__()

        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)


def disable_grad(net):
    for param in net.parameters():
        param.requires_grad = False

def get_parameters(nets):
    ''' Receives a list of modules, extracts the params and merges them '''
    parameters = []
    for net in nets:
        parameters += list(net.parameters())
    return parameters

def eval(nets):
    ''' Receives a list of modules and sets them to eval mode '''
    for net in nets:
        net.eval()

def train(nets):
    ''' Receives a list of modules and sets them to training mode '''
    for net in nets:
        net.train()

def get_channels(obs_space, args):
    ''' Calculates the number of input channels. Useful for instantiating conv nets'''
    # Check how many frames will be stacked
    n_stack = args.stacked_frames if args.stack else 1
    # Then check for grayscale
    channels = 1 * n_stack if args.grayscale else obs_space[0] * n_stack
    return channels

class no_grad_in:
    def __init__(self, *args): #modules):
        """
        Context manager to pause gradients in specific modules
        :param args: modules where the gradients will be paused.
                      args can be a list of modules or directly the (non-list) modules themselves
        """
        # Single flattened list and check in case some of the args are not lists
        self.modules = [module for arg in args for module in (arg if isinstance(arg, list) else [arg])]
        # else:
        #     self.modules = args[0]
        # #self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_value, traceback):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


