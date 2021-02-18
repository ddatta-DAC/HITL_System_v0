import redis
import pickle
import os
import pandas as pd

try:
    os.system("redis-server --port 6666 &")
except:
    print('[INFO] Redis server already running')


class redisStore:
    redis_conn = redis.Redis(host='localhost', port=6666, db=0)

    def __init__(self, DATA_LOC, subDIR):
        redisStore.ingest_data(
            DATA_LOC,
            subDIR
        )
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

        def aux_store(row):
            ID_COL = 'PanjivaRecordID'
            _dict = pickle.dumps(row.to_dict())
            key = str(int(_dict[ID_COL]))
            redisStore.redis_conn.set(key, _dict)

        df.apply(aux_store, axis=1)
        return

    @staticmethod
    def fetch_data(
            key
    ):
        read_dict = redisStore.redis_conn.get(str(key))
        _dict = pickle.loads(read_dict)
        return _dict


redisStore(
    DATA_LOC='./../generated_data',
    subDIR='01_2016'
)