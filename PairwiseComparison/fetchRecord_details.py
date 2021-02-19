import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')

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
from redisStore import setup_Redis
from utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
idMapping_df = None
config = None
ID_COL = 'PanjivaRecordID'
mapping_dict = None
op_save_dir = None
redis_obj = None

def get_domain_dims(
        subDIR
):
    global DATA_LOC
    with open(os.path.join(DATA_LOC, '{}/domain_dims.pkl'.format(subDIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def setup_up(subDIR):
    global config
    global saved_model_dir
    global idMapping_df
    global emb_save_dir
    global mapping_dict
    global op_save_dir
    global DATA_LOC
    global redis_obj
    
    redis_obj = setup_Redis.redisStore
    setup_Redis.redisStore(
        DATA_LOC=DATA_LOC,
        subDIR=subDIR
    )
    
    redis_obj.ingest_pairWiseDist(
        data_dir='./../PairwiseComparison/pairWiseDist',
        subDIR=subDIR
    )
    
    return 

    emb_save_dir = config['emb_save_dir']
    Path(emb_save_dir).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(emb_save_dir, subDIR)).mkdir(exist_ok=True, parents=True)

    idMapping_df = pd.read_csv(
        os.path.join(DATA_LOC, subDIR, 'idMapping.csv'),
        index_col=None
    )
    op_save_dir = config['op_save_dir']
    Path(op_save_dir).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(op_save_dir, subDIR)).mkdir(exist_ok=True, parents=True)
    mapping_dict = {}

    for domain in set(idMapping_df['domain']):
        tmp = idMapping_df.loc[(idMapping_df['domain'] == domain)]
        serial_id = tmp['serial_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        mapping_dict[domain] = {
            k: v for k, v in zip(serial_id, entity_id)
        }
    return
# =============================
# Front end facing method
# =============================
# id :: PanjivaRecordID
# while this id is global ;
# =============================
def fetchRecord_details(
        id = None,
        subDIR = None,
        emb_source = 'AD'
):

    global redis_obj
    global ID_COL
    record_dict = redis_obj.fetch_data(key=str(int(id)))
    domain_dims = get_domain_dims(subDIR)
    result = []

    for pair in combinations(list(domain_dims.keys()),2):
        pair = sorted(pair)
        d1,d2 = pair[0],pair[1]
        e1 = int(record_dict[d1])
        e2 = int(record_dict[d2])
        key = 'pairWiseD_{}_{}_{}_{}_{}'.format(emb_source,d1,e1,d2,e2)
        r = redis_obj.fetch_data(key)
        result.append(({d1:e1}, {d2:e2},float(r.decode())))

    return result



DATA_LOC ='./../generated_data_v1/us_import'
subDIR='01_2016'
setup_up(subDIR)
r = fetchRecord_details(
    id = 121983692,
    subDIR='01_2016'
)
print(r)