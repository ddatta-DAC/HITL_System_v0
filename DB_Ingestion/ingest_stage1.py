import yaml
import os
import pandas as pd
from sqlite_engine import sqlite
from sqlalchemy import create_engine
import pandas as pd
from joblib import Parallel,delayed
from sqlalchemy import inspect
from glob import glob
import multiprocessing as mp
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
from utils import hscode_filter_util

def _convert_to_int(x):
    try:
        return int(float(x))
    except:
        pass

    
def process_file(file_path):
    
    with open('config.yaml','r') as fh:
        config = yaml.safe_load(fh) 
    
    df = pd.read_csv(file_path, engine='c', low_memory=False, index_col=None, encoding='utf-8')
    df = df.dropna(subset=['HSCode'])

    df['HSCode'] = df['HSCode'].parallel_apply(
        hscode_filter_util.HSCode_filter_aux
    )

    df = df.dropna(subset=['HSCode'])
    print('HSCode filtered dataframe length >>', len(df))
    redundant_columns = []
    
    def process(column_group):
    
        _redundant_columns = []
        sqlite_conn = sqlite().get_engine()
        table_name = column_group['primary_key']
        primary_key = column_group['primary_key']
        columns = column_group['columns']

        _df_ = df[columns]
        _df_ = _df_[columns]
        _df_ = _df_.dropna(subset=[primary_key])
        if column_group['primary_key_type'] == 'int':
            _df_[primary_key] = _df_[primary_key].apply(_convert_to_int)
            _df_ = _df_.dropna(subset=[primary_key])
        else:
            _df_[primary_key] = _df_[primary_key].astype(column_group['primary_key_type'])
        
        _df_ = _df_.drop_duplicates()
        if 'filter_values' in column_group.keys():
            _remove = column_group['filter_values']
            _df_ = _df_.loc[~_df_[primary_key].isin(_remove) ]
            
        _df_.to_sql(
            table_name, 
            con=sqlite_conn,
            if_exists='append',
            index=False
        )
        
        print('Processed >>', table_name, len(_df_))
        _redundant_columns.extend(columns)
        _redundant_columns.remove(primary_key)
        return _redundant_columns
        
    res = Parallel(n_jobs = 10 ) (delayed(process)(column_group) for column_group in config['normalize'])
    for r in res: 
        redundant_columns.extend(r)
    
    valid_columns = [ _ for _ in list(df.columns) if _ not in redundant_columns]
    normalized_df = df[valid_columns]

    record_id = config['record_id']['name']
    record_id_type =  config['record_id']['type']

    if record_id_type == 'int':
        normalized_df.loc[:,record_id] = normalized_df[record_id].apply(_convert_to_int)
        normalized_df = normalized_df.dropna(subset=[record_id])

    for column_group in config['normalize']:   
        primary_key = column_group['primary_key']

        if column_group['primary_key_type'] == 'int':
            normalized_df.loc[:,primary_key] = normalized_df[primary_key].apply(_convert_to_int)
            normalized_df = normalized_df.dropna(subset=[primary_key])
        else:
            normalized_df.loc[:,primary_key] = normalized_df[primary_key].astype(column_group['primary_key_type'])
        if 'filter_values' in column_group.keys():
            _remove = column_group['filter_values']
            normalized_df = normalized_df.loc[~normalized_df[primary_key].isin(_remove) ]
            
    sqlite_conn = sqlite().get_engine()
    print(file_path, '||  Number of records', len(normalized_df))
    normalized_df.to_sql(
            'Records', 
            con=sqlite_conn,
            if_exists='append',
            index=False
    )
    return 



# =================================================================

def post_process():
    sqlite_conn = sqlite().get_engine()
    
    with open('config.yaml','r') as fh:
        config = yaml.safe_load(fh)
    
    for _group in  config['normalize']:
        table_name = _group['primary_key']
        df = pd.read_sql("select * from {}".format(table_name), sqlite_conn)
        df = df.drop_duplicates()
        _key = table_name
        # Add in ID column
        df =  df.sort_values(by=[_key])
        df['ID'] = np.arange(1, df.shape[0]+1)
        df.to_sql(
            table_name, 
            con=sqlite_conn,
            if_exists='replace',
            index=False
        )
    return


# =================================================================
with open('config.yaml','r') as fh:
    config = yaml.safe_load(fh)
    
DATA_LOC = config['file_path']
file_list = list(sorted( glob(os.path.join(DATA_LOC,'**.csv'))))
# Parallel(n_jobs = mp.cpu_count())( delayed(process_file)(file) for file in tqdm(file_list[:5]))
for file in tqdm(file_list[:]):
    process_file(file)

# ====================================================
# Post-process to eliminate duplicates
# ====================================================
post_process()

