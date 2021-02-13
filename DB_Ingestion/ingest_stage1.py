import yaml
import pandas as pd
from sqlite_engine import sqlite
from sqlalchemy import create_engine
import pandas as pd
from joblib import Parallel,delayed
sqlite_conn = sqlite().get_engine()
from glob import glob
import multiprocessing as mp
from tqdm import tqdm

def _convert_to_int(x):
    try:
        return int(float(x))
    except:
        pass

    
def process_file(file_path):
    with open('config.yaml','r') as fh:
        config = yaml.safe_load(fh) 
    
    df = pd.read_csv('tmp.csv', engine='python', index_col=None)
    redundant_columns = []
    for column_group in config['normalize']:   
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
        _df_.to_sql(
            table_name, 
            con=sqlite_conn,
            if_exists='replace',
            index=False
        )
        redundant_columns.extend(columns)
        redundant_columns.remove(primary_key)

    valid_columns = [_ for _ in list(df.columns) if _ not in redundant_columns]

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

    normalized_df.to_sql(
            'Records', 
            con=sqlite_conn,
            if_exists='replace',
            index=False
    )
    return 


file_list = list(sorted(os.path.join(DATA_LOC, glob('**.csv'))))
Parallel(n_jobs = mp.cpu_count())( delayed(file) for file in tqdm(file_list[:10]))


