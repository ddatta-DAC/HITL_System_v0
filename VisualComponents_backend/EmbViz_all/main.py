import pandas as pd
import numpy as np
import sys
sys.path.append('./../..')
from redisStore import redisUtil

DATA_LOC = None
subDIR = None
# Singleton object
redis_obj = redisUtil.redisStore

redisUtil.redisStore.ingest_record_data(
    DATA_LOC=DATA_LOC,
    subDIR=subDIR
)

redis_obj.ingest_MP2V_embeddings(
    DATA_LOC,
    subDIR ,
    mp2v_emb_dir='./../records2graph/saved_model_data',
    emb_dim=64
)










