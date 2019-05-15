# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np

def get_trajectory(thetas=None, radius=None, density=20, noise_std=0.02):
    if thetas is None:
        a = 2*np.pi*np.random.rand()
        b = 2*np.pi*(np.random.rand()*.7 + 0.3)
        thetas = [a, a+b]
    if radius is None:
        radius = np.random.rand()*.9 + 0.1
    
    circumference = 2*np.pi*radius * (thetas[1]-thetas[0])/(2*np.pi)
    samples = int(density*circumference)

    t = np.linspace(thetas[0],thetas[1], samples)
    x = np.cos(t)*radius + np.random.randn(*t.shape)*noise_std
    y = np.sin(t)*radius + np.random.randn(*t.shape)*noise_std
    
    dx = -y
    dy = x
    return x, y, dx, dy, t

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs)

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    b, a = b.flatten(), a.flatten()
    da = -b
    db = a

    field['x'] = np.stack( [a, b]).T
    field['dx'] = np.stack( [da, db]).T
    return field