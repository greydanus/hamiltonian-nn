# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLPAutoencoder, MLP
from hnn import HNN, PixelHNN
from data import get_dataset
from utils import L2_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=784, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=2, type=int, help='latent dimension of autoencoder')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--input_noise', default=0.0, type=int, help='std of noise added to HNN inputs')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=3000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=250, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='pendulum', type=str, help='either "real" or "sim" data')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim,
                               nonlinearity='relu')
  model = PixelHNN(args.latent_dim, args.hidden_dim,
                   autoencoder=autoencoder, nonlinearity=args.nonlinearity,
                   baseline=args.baseline)
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-6)

  # get dataset
  data = get_dataset(args.name, args.save_dir, verbose=True, seed=args.seed)
  trials = data['meta']['trials']
  timesteps = data['meta']['timesteps']
  inputs = torch.tensor( data['pixels'], dtype=torch.float32)
  inputs_next = torch.tensor( data['next_pixels'], dtype=torch.float32)

  # vanilla ae train loop
  for step in range(args.total_steps+1):
    # select a batch
    ixs = torch.randperm(trials*timesteps-2*trials)[:args.batch_size]
    x = inputs[ixs]
    x_next = inputs_next[ixs]

    # encode pixel space -> latent dimension
    z = model.encode(x)
    z_next = model.encode(x_next)

    # autoencoder loss
    x_hat = model.decode(z)
    ae_loss = L2_loss(x, x_hat)

    # hnn vector field loss
    noise = args.input_noise * torch.randn(*z.shape)
    z_hat_next = z + model.time_derivative(z + noise) # replace with rk4
    hnn_loss = L2_loss(z_next, z_hat_next)

    # canonical coordinate loss
    # -> makes latent space look like (x, v) coordinates
    w, dw = z.split(1,1)
    w_next, _ = z_next.split(1,1)
    cc_loss = L2_loss(dw, w_next - w)

    # sum losses and take a gradient step
    loss = cc_loss + ae_loss + 1e-2 * hnn_loss
    loss.backward() ; optim.step() ; optim.zero_grad()

    if args.verbose and step % 250 == 0:
      print("step {}, loss {:.4e}".format(step, loss.item()))

  return model

if __name__ == "__main__":
    args = get_args()
    model = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    path = '{}/{}-pixels-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)