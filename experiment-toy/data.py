# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np

def circular_vector_field(t, x):
  dx = np.fliplr(x.copy().reshape(-1,2))
  dx[:,0] *= -1
  return dx.copy().reshape(*x.shape)

def get_dataset(seed=0, xmin=-2, xmax=2, ymin=-2, ymax=2, noise_std=0, samples=400, test_split=0.5):
  data = {'meta': locals()}

  # randomly sample inputs
  np.random.seed(seed)
  a = np.random.rand(samples)*(ymax-ymin) + ymin
  b = np.random.rand(samples)*(xmax-xmin) + xmin
  
  # make vector field
  data['x'] = np.stack( [a, b]).T
  data['dx'] = circular_vector_field(t=None, x=data['x'])
  data['dx'] += noise_std * np.random.randn(*data['x'].shape)

  # make a train/test split
  split_ix = int(len(data['x']) * test_split)
  split_data = {}
  for k in ['x', 'dx']:
      split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
  data = split_data
  return data

def get_field(xmin=-2, xmax=2, ymin=-2, ymax=2, gridsize=20):
  field = {'meta': locals()}

  # meshgrid to get vector field
  b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
  b, a = b.flatten(), a.flatten()
  da = -b
  db = a
  
  field['x'] = np.stack( [a, b]).T
  field['dx'] = np.stack( [da, db]).T
  return field