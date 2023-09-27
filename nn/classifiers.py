import torch.nn as nn

class BilinearClassifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.net = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.net(x1, x2).squeeze(-1)