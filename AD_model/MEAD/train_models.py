import os
import torch
import argparse
import pandas as pd
import numpy as np

try:
    from . import model_AD_1 as AD
except:
    import model_AD_1 as AD
import pickle
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import sys

sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')
import yaml 

# ===================================== #

def get_domain_dims():
    global DATA_LOC
    global DIR
    with open( os.path.join(DATA_LOC,'{}/domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def get_training_data():
    global DATA_LOC
    global DIR
    x_pos = np.load(os.path.join(DATA_LOC,'{}/stage_2/train_x_pos.npy'.format(DIR)))
    x_neg = np.load(os.path.join(DATA_LOC,'{}/stage_2/train_x_neg.npy'.format(DIR)))
    return x_pos, x_neg


# ===================================== #

def main(
        subDIR,
        lr=None,
        batch_size=None,
        emb_dim=None,
        epochs=None,
        error_tol=0.01
):

    saved_model_dir = 'saved_model'
    path_obj = Path(saved_model_dir)
    path_obj.mkdir( parents=True, exist_ok=True)

    x_pos, x_neg = get_training_data()
    x_neg = x_neg.reshape([x_pos.shape[0], -1, x_pos.shape[1]])
    domain_dims = get_domain_dims()
    total_entity_count = sum(domain_dims.values())
    model = AD.AD_model_container(total_entity_count, emb_dim=emb_dim, device=device, lr= lr)
    model.train_model(x_pos, x_neg, batch_size=batch_size, epochs=epochs, tol = error_tol)
    model.save_model(os.path.join(saved_model_dir, subDIR))
    return


# ===================================== #
with open('config.yaml','r') as fh:
    config = yaml.safe_load(fh)
      
emb_dims = [int(_) for _ in config['emb_dims']]
DATA_LOC = config['DATA_LOC']
train_epochs = config['train_epochs']
learning_rate = float(config['learning_rate'])
batch_size = int(config['batch_size'])
error_tol = float(config['error_tol'])
    
with open(os.path.join(DATA_LOC, 'epoch_fileList.json'), 'r') as fh:
    epoch_fileList = json.load(fh)
    
subDIR_list = sorted(list(epoch_fileList.keys()))
      
        
for subDIR in subDIR_list:
    Parallel(n_jobs = MP.cpu_count())(
        delayed(main)(subDIR, learning_rate, batch_size, emb_dim, train_epochs, error_tol) for emb_dim in emb_dims
    )


