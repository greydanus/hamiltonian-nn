# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy.integrate
from orbits import custom_init_2d,update_fn,get_accelerations

def get_dataset(seed=0, xmin=-2, xmax=2, ymin=-2, ymax=2, noise_std=.5, samples=20):
  data = {'meta': locals()}

  #generate some orbits for training / testing split
  orbits=[]
  T=100
  t = np.linspace(0,T,100)
  for idx in range(samples):
    #one for training
    state = custom_init_2d(same_mass=True).flatten()
    solution = scipy.integrate.solve_ivp(update_fn, (0,T), state, t_eval=t, rtol=1e-14) # dense_output=True) #, t)
    trajectory = solution.y.T.reshape(solution.y.T.shape[0], -1, 5)
    forces = []
    for instant in trajectory:
      forces.append(get_accelerations(instant)[None,:,:])
    forces=np.vstack(forces)
    orbits.append({'initial':state,'trajectory':trajectory,'forces':forces})
    

  data['x']=np.vstack([ orbit['trajectory'] for orbit in orbits ])[:,:,:3].reshape(-1,6)
  data['dx']=np.vstack([ orbit['forces'] for orbit in orbits ]).reshape(-1,4)
 
  #try to normalize
  #data['x'][:,4:6]-=data['x'][:,1:3]
  #data['x'][:,1:3]=0
 
  #shuffle data before returning
  permutation = np.random.permutation(data['x'].shape[0])
  data['x']=data['x'][permutation]
  data['dx']=data['dx'][permutation]
  
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

