# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np

class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

  def forward(self, x, separate_fields=False):
    h = self.linear1(x).tanh()
    h = self.linear2(h).tanh()
    return self.linear3(h)

class MLPAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''
  def __init__(self, input_dim, hidden_dim, latent_dim):
    super(MLPAutoencoder, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.linear7, self.linear8]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

  def encoder(self, x):
    h = self.linear1(x).tanh()
    h = h + self.linear2(h).tanh()
    h = h + self.linear3(h).tanh()
    return self.linear4(h)

  def decoder(self, z):
    h = self.linear5(z).tanh()
    h = h + self.linear6(h).tanh()
    h = h + self.linear7(h).tanh()
    return self.linear8(h)

  def forward(self, x):
    z = self.encoder(x)
    x_hat = self.decoder(z)
    return x_hat