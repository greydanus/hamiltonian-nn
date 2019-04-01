# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import os
import numpy as np
from urllib.request import urlretrieve

from utils import read_lipson, str2array

def get_dataset(experiment_name, save_dir):
  '''Downloads and formats the datasets provided in the supplementary materials of
  the 2009 Lipson Science article "Distilling Free-Form Natural Laws from
  Experimental Data."
  Link to supplementary materials: https://bit.ly/2JNhyQ8
  Link to article: https://bit.ly/2I2TqXn
  '''
  if experiment_name == "sim":
    dataset_name = "pendulum_h_1"
  elif experiment_name == "real":
    dataset_name = "real_pend_h_1"
  else:
    assert experiment_name in ['sim', 'real']

  url = 'http://science.sciencemag.org/highwire/filestream/590089/field_highwire_adjunct_files/2/'
  os.makedirs(save_dir) if not os.path.exists(save_dir) else None
  out_file = '{}/invar_datasets.zip'.format(save_dir)
  
  urlretrieve(url, out_file)

  data_str = read_lipson(dataset_name, save_dir)
  data, names = str2array(data_str)
  return data, names