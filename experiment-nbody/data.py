# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.integrate
from orbits import custom_init_2d,update_fn,get_accelerations

def get_dataset(seed=0, xmin=-2, xmax=2, ymin=-2, ymax=2, noise_std=.5, samples=100, plot=False):
  data = {'meta': locals()}

  #generate some orbits for training / testing split
  orbits=[]
  T=20
  n=50
  t = np.linspace(0,T,n)
  for idx in range(samples):
    state = custom_init_2d(same_mass=True).flatten()
    trajectory = state.reshape(1,2,5)
    #trajectory = trajectory[1].reshape(1,2,5)
    if n>1:
      solution = scipy.integrate.solve_ivp(update_fn, (0,T), state, t_eval=t, rtol=1e-14) # dense_output=True) #, t)
      trajectory = solution.y.T.reshape(solution.y.T.shape[0], -1, 5)
    dxs = []
    for instant in trajectory:
      accelerations=get_accelerations(instant) #[nbodies x  2 (a_x, a_y)]
      #state = n x [mass, px, py, vx, vy]
      #dx = n x [0, vx, vy, ax, ay]
      dx = instant.copy()
      dx[:,0]=0
      dx[:,1:3]=dx[:,3:5]
      dx[:,3:5]=accelerations
      dxs.append(dx[None,:,:])
    dxs=np.vstack(dxs)
    orbits.append({'initial':state,'trajectory':trajectory,'dxs':dxs})
    
  data['x']=np.vstack([ orbit['trajectory'].reshape(-1,10) for orbit in orbits ]) #[:,:,:3].reshape(-1,6)
  data['dx']=np.vstack([ orbit['dxs'].reshape(-1,10) for orbit in orbits ]) #.reshape(-1,4)
 
  #try to normalize
  #data['x'][:,4:6]-=data['x'][:,1:3]
  #data['x'][:,1:3]=0

  #shuffle data before returning
  #permutation = np.random.permutation(data['x'].shape[0])
  #data['x']=data['x'][permutation]
  #data['dx']=data['dx'][permutation]

  if plot and len(plot)>0:
    #lets plot the orbits to this output folder
    os.makedirs(plot, exist_ok=True)
    for idx in range(len(orbits)):
      orbit = orbits[idx]
      plot_trajectory(orbit['trajectory'] , "%s/%d.png" % (plot,idx))

  return data


def plot_trajectory(trajectory, output_file):
  fig = plt.figure(figsize=[5,5], frameon=True, dpi=100)
  
  # draw trajectoryectories
  plt.title('Trajectories')
  colors = ['r', 'g', 'b']
  for i in range(trajectory.shape[1]):
      plt.plot(trajectory[:,i,1], trajectory[:,i,2], c=colors[i], label='body {} orbital'.format(i))
      plt.plot(trajectory[-1,i,1], trajectory[-1,i,2], 'o', c=colors[i])
  
  ax = fig.gca()
  ax.set_aspect(1)
  plt.xlabel('$x$') ; plt.ylabel('$y$')
  plt.legend(fontsize=6)
  plt.savefig(output_file)
  plt.close()

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

