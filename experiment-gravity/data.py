# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy
solve_ivp = scipy.integrate.solve_ivp

##### ENERGY #####
def potential_energy(state):
  '''U=\sum_i,j>i G m_i m_j / r_ij'''
  tot_energy = np.zeros((1,1,state.shape[2]))
  for i in range(state.shape[0]):
    for j in range(i+1,state.shape[0]):
      r_ij = ((state[i:i+1,1:3] - state[j:j+1,1:3])**2).sum(1, keepdims=True)**.5
      m_i = state[i:i+1,0:1]
      m_j = state[j:j+1,0:1]
      tot_energy += m_i * m_j / r_ij
  U = -tot_energy.sum(0).squeeze()
  return U

def kinetic_energy(state):
  '''T=\sum_i .5*m*v^2'''
  energies = .5 * state[:,0:1] * (state[:,3:5]**2).sum(1, keepdims=True)
  T = energies.sum(0).squeeze()
  return T

def total_energy(state):
  return potential_energy(state) + kinetic_energy(state)


##### DYNAMICS #####
def get_accelerations(state, epsilon=0):
  # shape of state is [bodies x properties]
  net_accs = [] # [nbodies x 2]
  for i in range(state.shape[0]): # number of bodies
    other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
    displacements = other_bodies[:, 1:3] - state[i, 1:3] # indexes 1:3 -> pxs, pys
    distances = (displacements**2).sum(1, keepdims=True)**0.5
    masses = other_bodies[:, 0:1] # index 0 -> mass
    pointwise_accs = masses * displacements / (distances**3 + epsilon) # main equation, G=1
    net_acc = pointwise_accs.sum(0, keepdims=True)
    net_accs.append(net_acc)
  net_accs = np.concatenate(net_accs, axis=0)
  return net_accs
  
def update(t, state):
  state = state.reshape(-1,5) # [bodies, properties]
  deriv = np.zeros_like(state)
  deriv[:,1:3] = state[:,3:5] # dx, dy = vx, vy
  deriv[:,3:5] = get_accelerations(state)
  return deriv.reshape(-1)


def random_config(nbodies=2):
  state = np.zeros((2,5))
  state[:,0] = 1
  pos = np.random.rand(2) + 0.5
  r = np.sqrt( np.sum((pos**2)) )
  
  # velocity that yields a circular orbit
  vel = np.flipud(pos) / (2 * r**1.5)
  vel[0] *= -1
  vel *= 1 + 5e-2*np.random.randn()
  
  # make the circular orbits SLIGHTLY elliptical
  state[:,1:3] = pos
  state[:,3:5] = vel
  state[1,1:] *= -1
  return state

def get_orbit(state, update_fn=update, t_points=100, t_span=[0,2], **kwargs):
  if not 'rtol' in kwargs.keys():
    kwargs['rtol'] = 1e-9
    
  orbit_settings = locals()
    
  nbodies = state.shape[0]
  t_eval = np.linspace(t_span[0], t_span[1], t_points)
  orbit_settings['t_eval'] = t_eval
  
  path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(), t_eval=t_eval, **kwargs)
  orbit = path['y'].reshape(nbodies, 5, t_points)
  return orbit, orbit_settings

# sample entire trajectories
def get_dataset(seed=0,samples=200,test_split=0.5):
  np.random.seed(seed)
  x, dx, e = [], [], []
  
  while len(x) < samples:
  
    state = random_config(nbodies=2)
    orbit, settings = get_orbit(state, t_points=50, t_span = [0, 20], rtol = 1e-9)
    batch = orbit.transpose(2,0,1).reshape(-1,10)
    for s in batch:
      dstate = update(None, s)
      x.append(s.reshape(-1,5)[:,1:].flatten())
      dx.append(dstate.reshape(-1,5)[:,1:].flatten())
      
      s = s.copy().reshape(2,5,1)
      e.append(total_energy(s))

  data={'x':np.stack(x), 'dx':np.stack(dx), 'e':np.stack(e)[...,None]}

  # make a train/test split
  split_ix = int(len(data['x']) * test_split)
  split_data = {}
  for k in ['x', 'dx','e']:
      split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
  data = split_data
  return data
