# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np

class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal'):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    def time_derivative(self, x, separate_fields=False):
        '''THIS IS WHERE THE MAGIC HAPPENS'''
        F1, F2 = self.forward(x) # traditional forward pass

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    @staticmethod
    def permutation_tensor(n):
        '''Constructs the Levi-Civita permutation tensor'''
        M = torch.ones(n,n) # matrix of ones
        M *= 1 - torch.eye(n) # clear diagonals
        M[::2] *= -1 # pattern of signs
        M[:,::2] *= -1

        for i in range(n): # make asymmetric
            for j in range(i+1, n):
                M[i,j] *= -1
        return M

class Baseline(torch.nn.Module):
    '''Wraps a baseline model so that it has the same basic API'''
    def __init__(self, input_dim, baseline_model):
        super(Baseline, self).__init__()
        self.baseline_model = baseline_model

    def forward(self, x):
        return self.baseline_model(x)

    def time_derivative(self, x):
        return self.forward(x)