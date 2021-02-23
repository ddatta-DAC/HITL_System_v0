import pandas as pd
import numpy as np
import sys
sys.path.append('./../..')
from redisStore import redisUtil

DATA_LOC = None
subDIR = None
# Singleton object
redis_obj = redisUtil.redisStore
EMB_DIM = None

# =============================
#  This needs to be called to ingest data into redis,
# =============================
def initialize(
    _DATA_LOC,
    _subDIR,
    mp2v_emb_dir = './../records2graph/saved_model_data',
    emb_dim = 64
):
    global redis_obj
    global DATA_LOC
    global subDIR
    global EMB_DIM

    EMB_DIM = emb_dim
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR

    redisUtil.redisStore.ingest_record_data(
        DATA_LOC=DATA_LOC,
        subDIR=subDIR
    )

    redis_obj.ingest_MP2V_embeddings(
        DATA_LOC,
        subDIR ,
        mp2v_emb_dir,
        emb_dim=emb_dim
    )


def get_record_entityEmbeddings(
    PanjivaRecordID
):
    record = redis_obj.fetch_data(
        str(int(PanjivaRecordID))
    )
    print(record)
    data_dict = {}
    for _domain, entity_id in record.items():
        key = 'mp2v_{}_{}_{}'.format(EMB_DIM, _domain, entity_id)
        vec = redis_obj.fetch_np(key)
        if vec is not None:
            data_dict[_domain] = vec
    return data_dict

'''
Example:
main.initialize(
    _DATA_LOC='./../generated_data_v1/us_import',
    _subDIR='01_2016',
    mp2v_emb_dir = './../records2graph/saved_model_data',
    emb_dim = 64
)

main.get_record_entityEmbeddings(
    PanjivaRecordID ='121983692'
)
'''










