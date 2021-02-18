import os
import pandas as pd
import sys
import torch
import glob

sys.path.append('./..')
sys.path.append('./../..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
from MEAD import model_AD_1 as mead_model
import yaml
from utils import borda_count
import json

saved_model_dir = None
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
    pathobj = Path(OUTPUT_DIR)
    pathobj.mkdir(exist_ok=True, parents=True)
    pathobj = Path(os.path.join(OUTPUT_DIR, subDIR))
    pathobj.mkdir(exist_ok=True, parents=True)
    return


def main(
        subDIR
):
    global OUTPUT_DIR
    global config
    global ID_COL
    global DATA_LOC
    global device
    global saved_model_dir

    test_df = pd.read_csv(
        os.path.join(DATA_LOC, '{}/stage_2/test_serialized.csv'.format(DIR)),
        index_col=None
    )
    domain_dims = get_domain_dims()
    entity_count = sum(domain_dims.values())
    id_list_normal = test_df[ID_COL].values.tolist()
    del test_df[ID_COL]
    test_x = test_df.values

    # ----
    # Find the models trained for the current epoch
    # ----
    trained_model_list = glob.glob(os.path.join(saved_model_dir, subDIR, 'model_**'))

    list_individual_ranks = []

    for model_path in trained_model_list:
        emb_dim = int(os.path.split(model_path)[-1].split('_')[1])

        model_container_obj = mead_model.AD_model_container(
            emb_dim=emb_dim,
            device=device,
            entity_count=entity_count
        )

        model_container_obj.load_model(model_path)
        scores_normal = model_container_obj.predict(test_x)
        scores = scores_normal
        id_list = id_list_normal

        data = {'score': scores, 'PanjivaRecordID': id_list}
        df = pd.DataFrame(data)
        # Lower == anomaly
        df = df.sort_values(by='score')
        list_individual_ranks.append(df[ID_COL].values.tolist())

    print('-----')
    combined_rank = borda_count.borda_count(list_individual_ranks)
    combined_rank = combined_rank.rename(columns={'ID': ID_COL})

    df = combined_rank.sort_values(by='rank', ascending=False)
    df.to_csv(os.path.join(OUTPUT_DIR, subDIR, 'AD_output.csv'), index=None)


with open('ensemble_config.yaml', 'r') as fh:
    config = yaml.safe_load(fh)

DATA_LOC = config['DATA_LOC']
OUTPUT_DIR = config['ENSEMBLE_OP_DIR']
saved_model_dir = config['saved_model_dir']

with open(os.path.join(DATA_LOC, 'epoch_fileList.json'), 'r') as fh:
    epoch_fileList = json.load(fh)

subDIR_list = sorted(list(epoch_fileList.keys()))

for subDIR in tqdm(subDIR_list):
    setup_up(subDIR)
    main(subDIR)
