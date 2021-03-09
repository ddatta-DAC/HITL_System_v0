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
from redisStore import redisUtil
from utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
idMapping_df = None
config = None
ID_COL = 'PanjivaRecordID'
mapping_dict = None
op_save_dir = None
redis_obj = redisUtil.redisStore

def get_domain_dims(
        subDIR
):
    global DATA_LOC
    with open(os.path.join(DATA_LOC, '{}/domain_dims.pkl'.format(subDIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

# ===========================
# data_dir : where all the base data is stored
# pairWiseDist_data_dir:  where pairwise proximities are stored
# ============================
def initialize(
    data_dir, 
    pairWiseDist_data_dir,
    subDIR
):
    global config
    global saved_model_dir
    global idMapping_df
    global emb_save_dir
    global mapping_dict
    global op_save_dir
    global DATA_LOC
    global redis_obj
    
    DATA_LOC = data_dir

 
    redis_obj.ingest_record_data(
        DATA_LOC=DATA_LOC,
        subDIR=subDIR
    )
    
    redis_obj.ingest_pairWiseDist(
        data_dir = pairWiseDist_data_dir ,
        subDIR = subDIR
    )
    

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



# data_dir ='./../generated_data_v1/us_import'
# pairwise_data_dir = './../PairwiseComparison/pairWiseDist'
# subDIR='01_2016'
# initialize(data_dir, pairwise_data_dir, subDIR)
# r = fetchRecord_details(
#     id = 121983692,
#     subDIR='01_2016'
# )
# print(r)