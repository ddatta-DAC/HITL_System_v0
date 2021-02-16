# Script to generate data for AD model
# -------------------------------------------
# Anomaly Data is already serialized
# Anomalies location: generated_data_v1/generated_anomalies/__DIR__
# Positive : pos_anomalies.csv
# Negative : neg_anomalies.csv
import yaml 
from tqdm import tqdm
import os
import json
import sys
sys.path.append('../../..')
sys.path.append('./..')
import pandas as pd
try:
    from .utils import utils
except:
    from utils import utils
import numpy as np
from pathlib import Path
# ============================
ID_COL = 'PanjivaRecordID'


def process(DATA_LOC, subDIR):
    global ID_COL
    
    data_source_dir = os.path.join(DATA_LOC, subDIR)
    save_dir_stage_2 = os.path.join(DATA_LOC, subDIR, 'stage_2') 
    path = Path(save_dir_stage_2)
    path.mkdir(exist_ok=True,parents=True)

    train_df = pd.read_csv(os.path.join(data_source_dir, 'train_data.csv' ), index_col=None)
    attributes = list(train_df.columns)
    attributes.remove(ID_COL)
    train_df = train_df.drop_duplicates(subset=attributes)
    train_df = utils.convert_to_serializedID_format(target_df = train_df, DIR=subDIR, data_source_loc = DATA_LOC)
    
   
    train_df.to_csv(os.path.join(save_dir_stage_2,'train_serialized.csv'), index=None)
    print('Post dropping duplicates, size of train set', len(train_df))
    x_pos, x_neg = utils.generate_negative_samples(train_df, DIR=subDIR, data_source_loc=DATA_LOC,num_neg_samples=12)
    # Save training data
    np.save(os.path.join(save_dir_stage_2,'train_x_pos.npy'),x_pos)
    np.save(os.path.join(save_dir_stage_2,'train_x_neg.npy'),x_neg)
    # --------------------------------------
    test_normal_df = pd.read_csv(os.path.join(data_source_dir, 'test_data.csv' ), index_col=None)
    test_normal_df = utils.convert_to_serializedID_format(target_df = test_normal_df, DIR=subDIR, data_source_loc = DATA_LOC)
#     test_normal_df = test_normal_df.drop_duplicates(subset=attributes)
#     print('Post dropping duplicates, size of test set', len(test_normal_df))
    test_normal_df.to_csv(os.path.join(save_dir_stage_2,'test_serialized.csv'),index=None)
    return



with open('ensemble_config.yaml','r') as fh:
    config = yaml.safe_load(fh)

DATA_LOC = config['DATA_LOC']
  
with open(os.path.join(DATA_LOC, 'epoch_fileList.json'), 'r') as fh:
    epoch_fileList = json.load(fh)
    
subDIR_list = sorted(list(epoch_fileList.keys()))
for subDIR in tqdm(subDIR_list):
    process(DATA_LOC, subDIR)

