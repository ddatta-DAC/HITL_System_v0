# Read in the models first
import os
import torch
from torch import FloatTensor as FT
from tqdm import tqdm
from pathlib import Path
import yaml
import json
import glob
import pandas as pd
import sys
from scipy.spatial.distance import cosine
import pickle
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.append('./..')
from AD_model.MEAD.model_AD_1 import AD_model_container
from DB_Ingestion.sqlite_engine import sqlite
from utils import utils

SQL_conn = sqlite().get_engine()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idMapping_df = None
config = None
ID_COL = 'PanjivaRecordID'


def get_domain_dims(
        subDIR
):
    global DATA_LOC
    with open(os.path.join(DATA_LOC, '{}/domain_dims.pkl'.format(subDIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def setup_up(subDIR):
    global config
    global OUTPUT_DIR
    global saved_model_dir
    global idMapping_df
    pathobj = Path(OUTPUT_DIR)
    pathobj.mkdir(exist_ok=True, parents=True)
    pathobj = Path(os.path.join(OUTPUT_DIR, subDIR))
    pathobj.mkdir(exist_ok=True, parents=True)
    idMapping_df = pd.read_csv(
        os.path.join(DATA_LOC, subDIR, 'idMapping.csv'),
        index_col=None
    )
    op_save_dir = config['op_save_dir']
    mapping_dict = {}

    for domain in set(idMapping_df['domain']):
        tmp = idMapping_df.loc[(idMapping_df['domain'] == domain)]
        serial_id = tmp['serial_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        mapping_dict[domain] = {
            k: v for k, v in zip(serial_id, entity_id)
        }
    return


def convert_from_serialized(
        target_df
):
    global idMapping_df
    global mapping_dict

    def _aux(val, domain):
        return mapping_dict[domain][val]

    for domain in set(idMapping_df['domain']):
        if domain not in list(target_df.columns): continue
        target_df.loc[:, domain] = target_df[domain].apply(_aux, args=(domain,))
    return target_df


def get_pairwise_dist(subDIR):
    global DATA_LOC
    global idMapping_df
    global ID_COL
    global emb_save_dir
    global op_save_dir
    # read in the test and train data
    df_train = pd.read_csv(
        os.path.join(DATA_LOC, subDIR, 'train_data.csv'), index_col=None
    )
    df_test = pd.read_csv(
        os.path.join(DATA_LOC, subDIR, 'test_data.csv'), index_col=None
    )
    df = df_train.append(df_test, ignore_index=True)
    del df[ID_COL]
    df = df.drop_duplicates()
    domain_dims = get_domain_dims(subDIR)

    emb_file_list = glob.glob(os.path.join(emb_save_dir, subDIR, '**.npy'))
    emb_list = [np.load(_) for _ in emb_file_list]

    def aux_find_dist(row, dom1, dom2, emb_arr):
        return cosine(emb_arr[row[dom1]], emb_arr[row[dom2]])

    for item in combinations(list(domain_dims.keys()), 2):
        item = list(sorted(item))
        _df_ = df[item].drop_duplicates()
        emb = np.zeros()
        dist_cols = []
        for emb in emb_list:
            d_col = 'dist_{}'.format(emb.shape[1])
            dist_cols.append(d_col)
            _df_[d_col] = _df_.parallel_apply(
                aux_find_dist,
                axis=1,
                args=(item[0], item[1], emb,)
            )
        # Find average distance
        _df_['dist'] = _df_.parallel_apply(
            lambda _row: np.mean([_row[_] for _ in dist_cols]),
            axis=1
        )
        _df_ = _df_[item[0], item[1], 'dist']
        _df_unserialized = convert_from_serialized(
            _df_
        )

        # Save
        key = '_'.join(item)
        fname = 'pairWiseDist_{}.csv'.format(key)
        Path(os.path.join(op_save_dir,subDIR )).mkdir(exist_ok=True,parents=True)
        path = os.path.join(op_save_dir, subDIR, fname)
        _df_unserialized.to_csv(
            path, index=None
        )
    return

def precompute_distances(
        subDIR
):
    global OUTPUT_DIR
    global config
    global ID_COL
    global DATA_LOC
    global device
    global saved_model_dir
    global emb_save_dir

    domain_dims = get_domain_dims(subDIR)
    entity_count = sum(domain_dims.values())

    # ----
    # Find the models trained for the current epoch
    # ----
    trained_model_list = glob.glob(os.path.join(saved_model_dir, subDIR, 'model_**'))
    emb_save_dir = ''
    emb_obj_list = []
    for model_path in trained_model_list:
        emb_dim = int(os.path.split(model_path)[-1].split('_')[1])
        model_container_obj = AD_model_container(
            emb_dim=emb_dim,
            device=device,
            entity_count=entity_count
        )
        model_container_obj.load_model(model_path)
        emb = model_container_obj.model.emb.cpu().data.numpy()
        path = os.path.join(emb_save_dir, subDIR)
        pathobj = Path(path)
        pathobj.mkdir(exist_ok=True, parents=True)
        fname = 'emb_{}.npy'.format(emb_dim)
        np.save(
            os.path.join(path, fname), emb
        )
        emb_obj_list.append(emb)

    # ------------------------------

    return


# =============================================================

with open('ensemble_config.yaml', 'r') as fh:
    config = yaml.safe_load(fh)

DATA_LOC = config['DATA_LOC']
saved_model_dir = config['saved_model_dir']

with open(os.path.join(DATA_LOC, 'epoch_fileList.json'), 'r') as fh:
    epoch_fileList = json.load(fh)

subDIR_list = sorted(list(epoch_fileList.keys()))

for subDIR in tqdm(subDIR_list):
    setup_up(subDIR)
    precompute_distances(subDIR)
