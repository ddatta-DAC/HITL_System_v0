import redis
import pickle
import os
import pandas as pd
import glob
from pandarallel import pandarallel
pandarallel.initialize()
DATA_STORED = False
try:
    os.system("redis-server --port 6666 &")
except:
    print('[INFO] Redis server already running')
    DATA_STORED = False


class redisStore:
    redis_conn = redis.Redis(host='localhost', port=6666, db=0)

    def __init__(self, DATA_LOC=None, subDIR=None):
        global DATA_STORED
        if DATA_LOC is None or subDIR is None:
            return 
        if not DATA_STORED:
            redisStore.ingest_data(
                DATA_LOC,
                subDIR
            )
        
        DATA_STORED = True
        return

    @staticmethod
    def get_redis_conn():
        return redisStore.redis_conn

    @staticmethod
    def ingest_data(
            DATA_LOC,
            subDIR
    ):
        df_train = pd.read_csv(
            os.path.join(DATA_LOC, subDIR, 'train_data.csv'), index_col=None
        )
        df_test = pd.read_csv(
            os.path.join(DATA_LOC, subDIR, 'test_data.csv'), index_col=None
        )
        df = df_train.append(df_test, ignore_index=True)

        def aux_store1(row):
            ID_COL = 'PanjivaRecordID'
            _dict = pickle.dumps(row.to_dict())
            key = str(int(row[ID_COL]))
            redisStore.redis_conn.set(key, _dict)

        df.parallel_apply(aux_store1, axis=1)
        return

    # ==========================
    # Store pairwise distances that have been precomputed
    # ===========================
    def ingest_pairWiseDist(
            data_dir,
            subDIR,
            emb_source='AD'
        ):
        import numpy as np
        from scipy import stats
        stats.percentileofscore([1, 2, 3, 4], 3)
        files = glob.glob(os.path.join(data_dir, subDIR,'**.csv'))

        def aux_store2(row , domain1, domain2):
            _dict = pickle.dumps(row.to_dict())
            key = 'pairWiseD_{}_{}_{}_{}_{}'.format(emb_source, domain1, int(row[domain1]), domain2, int(row[domain2]))
#             print('>>>', key, str(row['percentile']))
            redisStore.redis_conn.set(key, str(row['percentile']))
            return 

        for file in files:
            _df = pd.read_csv(file, index_col=None)
            # Calculate the percentile values
            fname = os.path.split(file)[-1].split('.')[0]
            domain1, domain2 = fname.split('_')[1:3]
            values = _df['dist'].values
            _df['percentile'] = _df['dist'].parallel_apply(lambda x: stats.percentileofscore(values, x)/100)
#             print(_df.head(10))
            _df.parallel_apply(aux_store2, axis=1, args=( domain1, domain2,))
            


    @staticmethod
    def fetch_data(
            key
    ):
        print('Key:', key)
        result = redisStore.redis_conn.get(str(key))
        try:
            result = pickle.loads(result)
            return result
        except:
            return result

# redisStore(
#     DATA_LOC='./../generated_data_v1/us_import',
#     subDIR='01_2016'
# )
# engine =  redisStore()
# print(engine.fetch_data(key='121983692'))
