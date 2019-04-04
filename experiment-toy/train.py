# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from nn_models import MLP
from hnn import HNN, HNNBaseline
from data import get_dataset
from utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--input_noise', default=0.5, type=int, help='std of noise added to inputs')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='toy', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default='.', type=str, help='name of dataset')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")
  if args.baseline:
    nn_model = MLP(args.input_dim, args.hidden_dim, args.input_dim, nonlinearity=args.nonlinearity)
    model = HNNBaseline(args.input_dim, baseline_model=nn_model)
  else:
    nn_model = MLP(args.input_dim, args.hidden_dim, 2, nonlinearity=args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type='solenoidal')

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  # arrange data
  data, val_data = get_dataset(seed=args.seed), get_dataset(seed=args.seed+1)
  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  
  val_x = torch.tensor( val_data['x'], requires_grad=True, dtype=torch.float32)
  val_dxdt = torch.Tensor(val_data['dx'])

  # vanilla train loop
  stats = {'train_loss': [], 'val_loss': []}
  for step in range(args.total_steps+1):
    

    dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
    dxdt_hat += args.input_noise * torch.randn(*x.shape) # add noise, maybe
    loss = L2_loss(dxdt, dxdt_hat)
    loss.backward() ; optim.step() ; optim.zero_grad()
    
    # validation stats
    val_dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
    val_loss = L2_loss(val_dxdt, val_dxdt_hat)
    stats['train_loss'].append(loss.item())
    stats['val_loss'].append(val_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, val_loss {:.4e}".format(step, loss.item(), val_loss.item()))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)