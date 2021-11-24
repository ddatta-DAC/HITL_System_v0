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
import multiprocessing as MP
from joblib import Parallel, delayed 
from tqdm import tqdm 
from glob import glob 

# ------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
idMapping_df = None
config = None
ID_COL = 'PanjivaRecordID'
mapping_dict = None
op_save_dir = None
redis_obj = redisUtil.redisStore
DATA_LOC = None
anomaly_result_dir = None
result_pkl_cacheDir = None
subDIR = None

# -------------------------------------------------------------------------

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
    _subDIR,
    _anomaly_result_dir,
    _result_pkl_cacheDir = 'pkl_Cache',
    load_redis = False
):
    global config
    global saved_model_dir
    global idMapping_df
    global emb_save_dir
    global mapping_dict
    global op_save_dir
    global DATA_LOC
    global redis_obj
    global anomaly_result_dir
    global result_pkl_cacheDir
    global subDIR
    
    DATA_LOC = data_dir
    subDIR = _subDIR
    anomaly_result_dir = _anomaly_result_dir
    print(_result_pkl_cacheDir, _subDIR)
    result_pkl_cacheDir = _result_pkl_cacheDir + '_' + _subDIR
    
    Path(result_pkl_cacheDir).mkdir(exist_ok=True, parents=True)
    
    redis_obj.ingest_record_data(
        DATA_LOC=DATA_LOC,
        subDIR = subDIR
    )
    print('Ingesting into redis ...')
    if load_redis:
        print('[Not loading REDIS!!!! ] can cause errors if results not cached')
        redis_obj.ingest_pairWiseDist(
            data_dir = pairWiseDist_data_dir,
            subDIR = subDIR
        )
    return

def setupGlobals(
   _DATA_LOC,
): 
    global DATA_LOC
    global subDIR
    DATA_LOC = _DATA_LOC
    return 

def __preload__(count=1000):
    
    global DATA_LOC, subDIR, anomaly_result_dir, ID_COL
    
    cpu_count = MP.cpu_count()
    # fetch the record ids
    ad_result = pd.read_csv(
        os.path.join(anomaly_result_dir, subDIR, 'AD_output.csv'), index_col=None
    )
    samples = ad_result.rename(columns={'rank': 'score'})
    samples = samples.sort_values(by='score', ascending=False)
    count = min(len(samples) // 3, count)
    df = samples.head(count).copy(deep=True)
    del df['score']
    recordID_list = df[ID_COL].tolist()
    
#     for id in tqdm(recordID_list):
#         get_stackedComparisonPlots(int(id))
    Parallel(n_jobs=cpu_count, prefer="threads")(
        delayed(fetchRecord_details)(int(id)) for id in tqdm(recordID_list)
    )
    return df


# =============================
# Front end facing method
# =============================
# id :: PanjivaRecordID
# while this id is global ;
# =============================
def fetchRecord_details(
        record_id = None,
        emb_source = 'AD'
):
    global subDIR, ID_COL, result_pkl_cacheDir
    file_signature  = '{}_pairwiseDist.pkl'.format(int(record_id))
    fpath = os.path.join(result_pkl_cacheDir, file_signature)
    
    if os.path.exists(fpath):
        print('Cached')
        with open(fpath,'rb') as fh:
            result = pickle.load(fh)
            return result
        
    redis_obj = redisUtil.redisStore
    ID_COL = 'PanjivaRecordID'
    record_dict = redis_obj.fetch_data(key=str(int(record_id)))
    
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
    result = list(sorted(result, key = lambda x: x[2], reverse=True))
    
    with open(fpath,'wb') as fh:
        pickle.dump(result, fh, pickle.HIGHEST_PROTOCOL)
    return result



# data_dir ='./../generated_data_v1/us_import'
# pairwise_data_dir = './../PairwiseComparison/pairWiseDist'
# subDIR='01_2016'
# initialize(data_dir, pairwise_data_dir, subDIR)
#
# r = fetchRecord_details(
#     id = 121983692,
#     subDIR='01_2016'
# )
# setupGlobals(
#   _DATA_LOC
# )
# print(r)
