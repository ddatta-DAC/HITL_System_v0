from sqlalchemy import create_engine
import pandas as pd
import os
import pickle
from sqlalchemy import Table, Column, Integer, String, MetaData
from pathlib import Path


class data_handler:
    def __init__(
            self,
            data_store_dir='./tmp',
            DATA_LOC=None,
            subDIR=None,
            anomaly_result_dir=None,
            anomaly_top_perc=5,
            anomaly_top_count=1000,
    ):
        Path(data_store_dir).mkdir(exist_ok=True, parents=True)
        ID_col = 'PanjivaRecordID'
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

        combined_df_1 = combined_df_1.sort_values(by='rank')

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
        # ---------------------------------------
        # Create a table for each domain
        # This is to store flagged entities
        # ---------------------------------------
        for dom in domains:
            self.create_domainTable(dom)

        if anomaly_top_count is None:
            anomaly_top_count = int( len(ad_result) * anomaly_top_perc / 100)

        # This is the count of records to be treated as potential anomalies
        self.anomaly_count = anomaly_top_count

        return

    # ------------------------------------
    # The domain table stores entities (specifically Consignee & Shipper) that have been flagged.
    # -------------------------------------
    def create_domainTable(self, domain):
        sql_query = """
        CREATE TABLE IF NOT EXISTS {} (
            {} BIGINT PRIMARY KEY,
            flag INT,   
        );""".format(domain, domain)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()
        return

    # ------------------------------------
    # The domain table stores entities (specifically Consignee & Shipper) that have been flagged.
    # -------------------------------------
    def store_entityFlag(self, domain, entity_id, flag=1):
        # Store the
        entity_value = self.domain_id2value_dict[domain][entity_id]

        sql_query = """ INSERT OR IGNORE INTO {} ({},{})
                      VALUES({} {}) """.format(domain, domain, 'flag', entity_value, flag)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()
        sql_query = "UPDATE {} SET flag ={} where {}={}".format(domain, flag, domain, entity_value)
        _cursor_ = self.sqllite_engine.cursor()
        _cursor_.execute(sql_query)
        _cursor_.commit()
        return

    def get_engine(self):
        return self.sqllite_engine
