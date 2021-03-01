from sqlalchemy import create_engine
import pandas as pd
import os
import pickle
from pathlib import Path
import itertools
import joblib
from joblib import Parallel, delayed
import multiprocessing as MP
from collections import OrderedDict
# ----------------------------------------
# Flag values:
# 0 : not set
# 1 : positive (relevant)
# -1 : negative (not relevant)
# ----------------------------------------
class data_handler:
    def __init__(
            self,
            DATA_LOC: str,
            subDIR: str,
            embedding_data_path: str,
            anomaly_result_dir: str,
            data_store_dir='./tmp',
            anomaly_top_perc=5,
            anomaly_top_count=1000

    ):
        ID_col = 'PanjivaRecordID'
        self.embedding_data_path = embedding_data_path
        # This table stores the records upon which update happens
        self.recordsTable = 'anomaly_records'
        self.DATA_LOC = DATA_LOC
        self.subDIR = subDIR
        self.anomaly_result_dir = anomaly_result_dir
        self.ID_col = ID_col
        self.entity_dict = {}
        Path(data_store_dir).mkdir(exist_ok=True, parents=True)

        # Read in column mapping
        self.domain_value2id_dict = None
        with open(os.path.join(DATA_LOC, subDIR, 'col_val2id_dict.pkl'), 'rb') as fh:
            self.domain_value2id_dict = pickle.load(fh)
        self.domain_id2value_dict = {}
        for dom in self.domain_value2id_dict.keys():
            self.domain_id2value_dict[dom] = {
                v: k for k, v in self.domain_value2id_dict[dom].items()
            }

        domains = list(self.domain_id2value_dict.keys())
        self.domain_dims = None
        with open(os.path.join(self.DATA_LOC, self.subDIR, 'domain_dims.pkl'), 'rb') as fh:
            self.domain_dims = OrderedDict(pickle.load(fh))

        # -----------------------------------------
        # This is a local sqllite3 database
        # -----------------------------------------
        self.sqllite_engine = create_engine(
            'sqlite:///{}/online_update.db'.format(data_store_dir),
            echo=False
        )

        # Read in the anomaly detection results
        ad_result = pd.read_csv(
            os.path.join(anomaly_result_dir, subDIR, 'combined_output.csv'),
            index_col=None
        )

        # Read in test data
        test_data = pd.read_csv(
            os.path.join(DATA_LOC, subDIR, 'test_data.csv'),
            index_col=None
        )

        # Join them
        combined_df_1 = test_data.merge(
            ad_result,
            on=ID_col,
            how='inner',
        )

        combined_df_1 = combined_df_1.rename(columns={'rank': 'score'})
        combined_df_1 = combined_df_1.sort_values(by='score', ascending=False)
        combined_df_1['flag'] = 0
        combined_df_1 = combined_df_1.sort_values(by='score')
        assert len(combined_df_1) == len(test_data) and len(combined_df_1) == len(ad_result), print(' Merge error !!!')

        # ---------------------------------------
        # Create a table for each domain
        # This is to store flagged entities
        # ---------------------------------------
        for dom in domains:
            self.create_domainTable(dom)

        if anomaly_top_count is None:
            anomaly_top_count = int(len(ad_result) * anomaly_top_perc / 100)

        # This is the count of records to be treated as potential anomalies
        self.anomaly_count = anomaly_top_count
        # Potential anomalies
        potential_anomalies = combined_df_1.head(self.anomaly_count)
        # Store in sqlite data base
        potential_anomalies.to_sql(
            self.recordsTable,
            con=self.sqllite_engine,
            if_exists='replace',
            index=False
        )
        self.data_df = potential_anomalies.copy(deep=True)
        del self.data_df['score']
        return

    def get_entity(self, recordID, domain):
        return self.data_df.loc[self.data_df[self.ID_col]==recordID][domain]

    def get_serialID_to_entityID(self):
        serialID_mapping_loc = os.path.join(self.DATA_LOC, self.subDIR, 'idMapping.csv')
        idMapper_file = os.path.join(serialID_mapping_loc)
        mapping_df = pd.read_csv(idMapper_file, index_col=None)
        serialID_to_entityID = {}

        for i, row in mapping_df.iterrows():
            serialID_to_entityID[row['serial_id']] = row['entity_id']
        return serialID_to_entityID

    # ---------------------------------
    # These are the records upon which actions are performed.
    # ---------------------------------
    def get_working_records(self, exclude_updated=True):
        sql_query = 'select * from {} ORDER BY score DESCENDING'.format(self.recordsTable)
        df = pd.read_sql(
            sql_query,
            con=self.sqllite_engine, index_col=None)

        if exclude_updated:
            df = df.loc[df['label'] == 0]
        return df

    # ------------------------------------
    # The domain table stores entities (specifically Consignee & Shipper) that have been flagged.
    # -------------------------------------
    def create_domainTable(self, domain):
        self.entity_dict[domain] = {}
        self.entity_dict[domain] = {k: 0 for k in self.domain_value2id_dict[domain].keys()}

        sql_query = """
        CREATE TABLE IF NOT EXISTS {} (
            {} BIGINT PRIMARY KEY,
            label INT,   
        );""".format(domain, domain)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()
        return

    # ------------------------------------
    # The domain table stores entities (specifically Consignee & Shipper) that have been flagged.
    # -------------------------------------
    def store_entityFlag(self, domain, entity_id, label=1) -> object:
        entity_value = self.domain_id2value_dict[domain][entity_id]
        self.entity_dict[domain][entity_value] = label
        sql_query = """ INSERT OR IGNORE INTO {} ({},{})
                      VALUES({} {}) """.format(domain, domain, 'label', entity_value, label)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()

        sql_query = "UPDATE {} SET label ={} where {}={}".format(domain, label, domain, entity_value)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()
        return

    def get_entityFlag(self, domain, entity_value=None, entity_id=None):
        assert entity_value is not None or entity_id is not None, print(
            '[Error!] check arguments passed: function::get_entityFlag')
        if entity_value is None:
            entity_value = self.domain_id2value_dict[domain][entity_id]
        return self.entity_dict[domain][entity_value]

    # ------------------------------------
    # The domain table stores entities (specifically Consignee & Shipper) that have been flagged.
    # -------------------------------------
    def store_recordFlag(self, recordID, label=1):
        sql_query = "UPDATE {} SET label = {} where {}={}".format('PanjivaRecordID', label, 'PanjivaRecordID', recordID)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()
        return

    def get_nominal_records_sample(self, count=5000):

        # Read in test data
        ad_result = pd.read_csv(
            os.path.join(self.anomaly_result_dir, self.subDIR, 'combined_output.csv'),
            index_col=None
        )

        # Read in test data
        test_data = pd.read_csv(
            os.path.join(self.DATA_LOC, self.subDIR, 'test_data.csv'),
            index_col=None
        )

        # Join them
        combined_df_1 = test_data.merge(
            ad_result,
            on=self.ID_col,
            how='inner',
        )
        samples = combined_df_1.rename(columns={'rank': 'score'})
        samples = samples.sort_values(by='score', ascending=True)
        count = min(len(samples) // 3, count)
        df = samples.head(count)
        del df['score']
        return df

    def get_engine(self):
        return self.sqllite_engine

    def create_entityInteractionStore(
            self
    ):
        domain_list = list(self.domain_dims.keys())
        self.entityInteraction_pairs = {}
        for domain_pair in itertools.combinations(domain_list,2):
            key = '_'.join(list(sorted(domain_pair)))
            self.entityInteraction_pairs[key] = []

    def register_entityInteraction(
            self,
            domain_1,
            entity_1,
            domain_2,
            entity_2
    ):
        if domain_2 < domain_1:
            domain_1,domain_2 = domain_2,domain_1
            entity_1,entity_2 = entity_2,entity_1

        key = '_'.join([domain_1,domain_2])
        val = (entity_1,entity_2)
        self.entityInteraction_pairs[key].append(val)
        return

    def get_domainPairInteractionsRegistered(self, recordID):
        record_row = self.data_df.loc[self.data_df[self.ID_col]==recordID]
        domain_list = list(self.domain_dims.keys())
        keys = []
        for domain_pair in itertools.combinations(domain_list, 2):
            keys.append('_'.join(list(sorted(domain_pair))))

        def aux(key):
            nonlocal  record_row
            d1,d2 = key.split('_')
            e1 = record_row[d1]
            e2 = record_row[d2]
            if (e1,e2) in self.entityInteraction_pairs[key]:
                return (key,True)
            else:
                return (key,False)
        results = Parallel(n_jobs=MP.cpu_count())(delayed(aux)(key) for key in keys)
        domain_pairs = [ _item[0] for _item in results if _item[1] == True ]
        return domain_pairs
