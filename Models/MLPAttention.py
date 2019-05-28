import torch.nn as nn
import torch.nn.functional as F
import torch


class MLPAttention(nn.Module):

    def __init__(self, hidden_dim, annotation_dim):
        super().__init__()



        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, Q, K):
