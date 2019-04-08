# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import os
import numpy as np
from urllib.request import urlretrieve

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import read_lipson, str2array

def get_dataset(experiment_name, save_dir, test_split=0.8):
  '''Downloads and formats the datasets provided in the supplementary materials of
  the 2009 Lipson Science article "Distilling Free-Form Natural Laws from
  Experimental Data."
  Link to supplementary materials: https://bit.ly/2JNhyQ8
  Link to article: https://bit.ly/2I2TqXn
  '''
  if experiment_name == "pend-sim":
    dataset_name = "pendulum_h_1"
  elif experiment_name == "pend-real":
    dataset_name = "real_pend_h_1"
  else:
    assert experiment_name in ['sim', 'real']

  url = 'http://science.sciencemag.org/highwire/filestream/590089/field_highwire_adjunct_files/2/'
  os.makedirs(save_dir) if not os.path.exists(save_dir) else None
  out_file = '{}/invar_datasets.zip'.format(save_dir)
  
  urlretrieve(url, out_file)

  data_str = read_lipson(dataset_name, save_dir)
  state, names = str2array(data_str)

  # put data in a dictionary structure
  data = {k: state[:,i:i+1] for i, k in enumerate(names)}
  data['x'] = state[:,2:4]
  data['dx'] = (data['x'][1:] - data['x'][:-1]) / (data['t'][1:] - data['t'][:-1])
  data['x'] = data['x'][:-1]

  # make a train/test split while preserving order of data
  split_ix = int(len(data['x']) * test_split)
  split_data = {}
  for k, v in data.items():
      split_data[k], split_data['test_' + k] = v[:split_ix], v[split_ix:]
  data = split_data
  return data
