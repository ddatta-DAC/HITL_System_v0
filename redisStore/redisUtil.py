import redis
import pickle
import os
import pandas as pd
import glob
from pandarallel import pandarallel
import numpy as np
from sklearn.manifold import TSNE
import msgpack
import msgpack_numpy as msgpk_np

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
    def ingest_record_data(
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
        files = glob.glob(os.path.join(data_dir, subDIR, '**.csv'))

        def aux_store2(row, domain1, domain2):
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
            _df['percentile'] = _df['dist'].parallel_apply(lambda x: stats.percentileofscore(values, x) / 100)
            _df.parallel_apply(aux_store2, axis=1, args=(domain1, domain2,))

    @staticmethod
    def fetch_data(
            key
    ):
        result = redisStore.redis_conn.get(str(key))
        try:
            result = pickle.loads(result)
            return result
        except:
            return result

    @staticmethod
    def fetch_np(
            key
    ):
        data = msgpk_np.unpackb(redisStore.redis_conn.get('d'))
        return data

    # ----------
    #  This data requires fetch_np
    # -----------
    @staticmethod
    def ingest_MP2V_embeddings(
            DATA_LOC,
            subDIR,
            mp2v_emb_dir,
            emb_dim=64
    ):
        file_list = glob.glob(os.path.join(mp2v_emb_dir, subDIR, 'mp2v_**{}**.npy'.format(emb_dim)))
        emb_arr_dict = {}
        for file in file_list:
            fname = os.path.split(file)[-1]
            domain = fname.split('_')[1]
            arr = np.load(file, allow_pickle=True)
            emb_arr_dict[domain] = arr
        # --------------------------
        # Convert to serial mapping
        # --------------------------
        idMapping_df = pd.read_csv(
            os.path.join(DATA_LOC, subDIR, 'idMapping.csv'), index_col=None
        )
        idMapping_df['serial_id'] = idMapping_df['serial_id'].astype(int)
        idMapping_df['entity_id'] = idMapping_df['entity_id'].astype(int)

        enityID2serialID_mapping_dict = {}
        serialIDentityID_mappping_dict = {}
        for domain in set(idMapping_df['domain']):
            tmp = idMapping_df.loc[(idMapping_df['domain'] == domain)]
            serial_id = tmp['serial_id'].values.tolist()
            entity_id = tmp['entity_id'].values.tolist()
            enityID2serialID_mapping_dict[domain] = {k: v for k, v in zip(entity_id, serial_id)}
            serialIDentityID_mappping_dict[serial_id] = (domain, serial_id)
        total_entity_count = len(idMapping_df)
        X = np.zeros([total_entity_count, emb_dim])
        for domain in set(idMapping_df['domain']):
            arr = emb_arr_dict[domain]
            e_id_list = idMapping_df.loc[idMapping_df['domain'] == domain]['entity_id']
            for entity_id in e_id_list:
                serial_id = enityID2serialID_mapping_dict[domain][entity_id]
                X[serial_id] = arr[entity_id]

        projections_dict = {}
        tsne = TSNE(n_components=2, random_state=0, n_iter=1500, learning_rate=10)
        projections = tsne.fit_transform(X)

        for domain in set(idMapping_df['domain']):
            e_id_list = idMapping_df.loc[idMapping_df['domain'] == domain]['entity_id']
            projections_dict[domain] = np.zeros([len(e_id_list), 2])
            for entity_id in e_id_list:
                serial_id = enityID2serialID_mapping_dict[domain][entity_id]
                projections_dict[domain][entity_id] = projections[serial_id]

        for domain in set(idMapping_df['domain']):
            e_id_list = idMapping_df.loc[idMapping_df['domain'] == domain]['entity_id']
            for entity_id in e_id_list:
                key = 'mp2v_{}_{}_{}'.format(emb_dim, domain, entity_id)
                data = projections_dict[domain][entity_id]
                data = msgpk_np.packb(data)
                redisStore.redis_conn.set(key, data)
        return


# redisStore(
#     DATA_LOC='./../generated_data_v1/us_import',
#     subDIR='01_2016'
# )
# engine =  redisStore()
# print(engine.fetch_data(key='121983692'))
