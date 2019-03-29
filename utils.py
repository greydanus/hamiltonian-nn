# Sam Greydanus, Misko Dzama, Jason Yosinski
# 2019 | Google AI Residency Project "Hamiltonian Neural Networks"

import os
import pickle
import zipfile
import numpy as np

def L2_loss(u, v):
  return (u-v).pow(2).mean()

def read_lipson(experiment_name, save_dir):
  desired_file = experiment_name + ".txt"
  with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_dir)) as z:
    for filename in z.namelist():
      if desired_file == filename and not os.path.isdir(filename):
        with z.open(filename) as f:
            data = f.read()
  return str(data)

def str2array(string):
  lines = string.split('\\n')
  names = lines[0].strip("b'% \\r").split(' ')
  dnames = ['d' + n for n in names]
  names = ['trial', 't'] + names + dnames
  data = [[float(s) for s in l.strip("' \\r,").split( )] for l in lines[1:-1]]

  return np.asarray(data), names


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing