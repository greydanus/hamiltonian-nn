# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, time, argparse
import numpy as np
import matplotlib.pyplot as plt

from data_toy import get_dataset, get_field
from nn_models import MLP
from hnn import HNN, HNNBaseline
from utils import L2_loss, integrate_model

LINE_SEGMENTS = 10
ARROW_SCALE = 30
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
  parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
  parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
  parser.add_argument('--gridsize', default=10, type=int, help='gridsize of vector field in plots')
  parser.add_argument('--dpi', default=300, type=int, help='resolution of plot')
  parser.add_argument('--format', default='pdf', type=str, help='format of plot')
  parser.add_argument('--name', default='toy', type=str, help='only one option right now')
  parser.add_argument('--seed', default=0, type=int, help='random seed')
  parser.add_argument('--save_dir', default='./saved', type=str, help='name of dataset')
  parser.add_argument('--fig_dir', default='./figures', type=str, help='name of dataset')
  parser.set_defaults(feature=True)
  return parser.parse_args()

def get_model(args, baseline):

  if baseline:
    nn_model = MLP(args.input_dim, args.hidden_dim, args.input_dim, nonlinearity=args.nonlinearity)
    model = HNNBaseline(args.input_dim, baseline_model=nn_model)
    path = "{}/toy-baseline.tar".format(args.save_dir)
    model.load_state_dict(torch.load(path))
  else:
    nn_model = MLP(args.input_dim, args.hidden_dim, 2, nonlinearity=args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type='solenoidal')
    path = "{}/toy-hnn.tar".format(args.save_dir)
    model.load_state_dict(torch.load(path))

  return model

def get_vector_field(model, **kwargs):
  field = get_field(**kwargs)
  np_mesh_x = field['x']

  mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32)
  mesh_dx = model.time_derivative(mesh_x)
  return mesh_dx.data.numpy()


if __name__ == "__main__":
  args = get_args()

  ###### LOAD AND ANALYZE ######
  # load models
  base_model = get_model(args, baseline=True)
  hnn_model = get_model(args, baseline=False)

  # get their vector fields
  field = get_field(gridsize=args.gridsize)
  data = get_dataset()
  base_field = get_vector_field(base_model, gridsize=args.gridsize)
  hnn_field = get_vector_field(hnn_model, gridsize=args.gridsize)

  # integrate along those fields starting from point (1,0)
  t_span = [0,50]
  y0 = np.asarray([1., 0])
  kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 1000)}
  base_ivp = integrate_model(base_model, t_span, y0, **kwargs)
  hnn_ivp = integrate_model(hnn_model, t_span, y0, **kwargs)


  ###### PLOT ######
  fig = plt.figure(figsize=(12, 3), facecolor='white', dpi=args.dpi)

  # plot vector field
  ax = fig.add_subplot(1, 4, 1, frameon=True)
  ax.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH)  
  ax.set_xlabel("$x_0$") ; ax.set_ylabel("$x_1$")
  ax.set_title("Vector field")

  # plot dataset
  ax = fig.add_subplot(1, 4, 2, frameon=True)
  ax.quiver(data['x'][:,0], data['x'][:,1], data['dx'][:,0], data['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH)  
  ax.set_xlabel("$x_0$") ; ax.set_ylabel("$x_1$")
  ax.set_title("Toy dataset")

  # plot baseline
  ax = fig.add_subplot(1, 4, 3, frameon=True)
  ax.quiver(field['x'][:,0], field['x'][:,1], base_field[:,0], base_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH)

  for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    ax.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

  ax.set_xlabel("$x_0$") ; ax.set_ylabel("$x_1$")
  ax.set_title("Baseline NN")

  # plot HNN
  ax = fig.add_subplot(1, 4, 4, frameon=True)
  ax.quiver(field['x'][:,0], field['x'][:,1], hnn_field[:,0], hnn_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH)
  
  for i, l in enumerate(np.split(hnn_ivp['y'].T, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    ax.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

  ax.set_xlabel("$x_0$") ; ax.set_ylabel("$x_1$")
  ax.set_title("Hamiltonian NN")

  plt.tight_layout() ; fig.savefig('{}/toy.{}'.format(args.fig_dir, args.format))