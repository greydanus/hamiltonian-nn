# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np

def get_dataset(seed=0, xmin=-2, xmax=2, ymin=-2, ymax=2, noise_std=.5, samples=200):
  data = {'meta': locals()}

  # random sample
  np.random.seed(seed)
  a = np.random.rand(samples)*(ymax-ymin) + ymin
  b = np.random.rand(samples)*(xmax-xmin) + xmin
  da = -b + noise_std * np.random.randn(samples)
  db = a + noise_std * np.random.randn(samples)
  
  data['x'] = np.stack( [a, b]).T
  data['dx'] = np.stack( [da, db]).T
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