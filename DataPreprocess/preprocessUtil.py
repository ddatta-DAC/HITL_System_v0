import pandas as pd
import numpy as np
import os
import sys
from collections import OrderedDict
sys.path.append('./../..')
sys.path.append('./..')
from DB_Ingestion.sqlite_engine import  sqlite
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from pandarallel import pandarallel
pandarallel.initialize()
import yaml
from collections import Counter
import pickle
sys.path.append('./..')
sys.path.append('./../..')
from utils import hscode_filter_util


CONFIG = None
CONFIG_FILE = 'config.yaml'
id_col = 'PanjivaRecordID'

use_cols = None
freq_bound = None
save_dir = None
attribute_columns = None

def set_up_config(__dir__):
    global CONFIG
    global CONFIG_FILE
    global use_cols
    global num_neg_samples_ape
    global save_dir
    global column_value_filters
    global num_neg_samples
    global freq_bound_PERCENTILE
    global freq_bound_ABSOLUTE
    global id_col
    global attribute_columns

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    save_dir = __dir__

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG['use_cols']

    freq_bound_ABSOLUTE = CONFIG['low_freq_bound']
    freq_bound_PERCENTILE = CONFIG['freq_bound_PERCENTILE']
    column_value_filters = CONFIG['column_value_filters']

    _cols = list(use_cols)
    _cols.remove(id_col)
    attribute_columns = list(sorted(_cols))
    return

def HSCode_cleanup(list_df):
    new_list = []
    for _df in list_df:
        _df['HSCode'] = _df['HSCode'].parallel_apply(hscode_filter_util.HSCode_filter_aux)
        _df = _df.dropna(subset=['HSCode'])
        print(' In HSCode clean up , length of dataframe ', len(_df))
        new_list.append(_df)
    return new_list

def remove_low_frequency_values(df):
    global id_col
    global freq_bound_PERCENTILE
    global freq_bound_ABSOLUTE
    global attribute_columns

    freq_column_value_filters = {}
    feature_cols = list(attribute_columns)
    print('feature columns ::', feature_cols)
    # ----
    # Figure out which entities are to be removed
    # ----

    counter_df = pd.DataFrame(columns=['domain', 'count'])
    for c in feature_cols:
        count = len(set(df[c]))
        counter_df = counter_df.append({
            'domain': c, 'count': count
        }, ignore_index=True)

        z = np.percentile(
            list(Counter(df[c]).values()), 5)
        print(c, count, z)

    counter_df = counter_df.sort_values(by=['count'], ascending=True)
    print(' Data frame of Number of values', counter_df)

    for c in list(counter_df['domain']):

        values = list(df[c])
        freq_column_value_filters[c] = []
        obj_counter = Counter(values)
        for _item, _count in obj_counter.items():
            if _count < freq_bound_PERCENTILE or _count < freq_bound_ABSOLUTE:
                freq_column_value_filters[c].append(_item)

    print('Removing :: ')
    for c, _items in freq_column_value_filters.items():
        print('column : ', c, 'count', len(_items))

    print(' DF length : ', len(df))
    for col, val in freq_column_value_filters.items():
        df = df.loc[~df[col].isin(val)]

    print(' DF length : ', len(df))
    return df

def apply_value_filters(list_df):
    global column_value_filters

    if type(column_value_filters) != bool:
        list_processed_df = []
        for df in list_df:
            for col, val in column_value_filters.items():
                df = df.loc[~df[col].isin(val)]
            list_processed_df.append(df)
        return list_processed_df
    return list_df

def _convert_to_int(x):
    try:
        return int(float(x))
    except:
        pass


def clean_train_data(
        train_files
):

    global CONFIG
    global attribute_columns
    global use_cols

    files = train_files
    print('Columns read ', use_cols)

    list_df = [
        pd.read_csv(
            _file,
            usecols=use_cols,
            low_memory=False
        ) for _file in files
    ]
    
    list_df = [_.dropna() for _ in list_df]
    list_df = HSCode_cleanup(list_df)

    list_df_1 = apply_value_filters(list_df)
    master_df = None
    for df in list_df_1:
        if master_df is None:
            master_df = pd.DataFrame(df, copy=True)
        else:
            master_df = master_df.append(
                df,
                ignore_index=True
            )
    master_df = remove_low_frequency_values(master_df)
    print(len(master_df))
    
    return master_df

def order_cols(df):
    global attribute_columns
    global id_col
    ord_cols = [id_col] + attribute_columns
    return df[ord_cols]

def convert_to_ids(
        df,
        save_dir
):
    global id_col
    global freq_bound
    global attribute_columns
    global CONFIG
    pandarallel.initialize()

#     for attr,_type in CONFIG['data_types'].items():
#         if _type == 'int':
#             df[attr] = df[attr].apply(_convert_to_int)
#             df = df.dropna(subset=[attr])

    feature_columns = list(sorted(attribute_columns))
    dict_DomainDims = {}
    col_val2id_dict = {}

    for col in feature_columns:
        vals = list(set(df[col]))
        vals = list(sorted(vals))

        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }
        print(' > ', col, ':', len(id2val_dict))

        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict

        # Replace
        df[col] = df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(col, val2id_dict,)
        )

        dict_DomainDims[col] = len(id2val_dict)

    print(' Feature columns :: ', feature_columns)
    print(' dict_DomainDims ', dict_DomainDims)

    # -------------
    # Save the domain dimensions
    # -------------

    file = 'domain_dims.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            dict_DomainDims,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    file = 'col_val2id_dict.pkl'
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            col_val2id_dict,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    return df, col_val2id_dict

def replace_attr_with_id(row, attr, val2id_dict):
    val = row[attr]
    if val not in val2id_dict.keys():
        return None
    else:
        return val2id_dict[val]

def setup_testing_data(
        test_df,
        col_val2id_dict,
        drop_duplicates = False
):
    global id_col
    global save_dir
    global CONFIG
    global attribute_columns
    test_df = test_df.dropna()
    
    for attr,_type in CONFIG['data_types'].items():
        if _type == 'int':
            test_df[attr] = test_df[attr].apply(_convert_to_int)
            test_df = test_df.dropna(subset=[attr])

    # Replace with None if ids are not in train_set
    feature_cols = list(attribute_columns)

    for col in feature_cols:
        valid_items = list(col_val2id_dict[col].keys())
        test_df = test_df.loc[test_df[col].isin(valid_items)]

    # First convert to to ids
    for col in feature_cols:
        val2id_dict = col_val2id_dict[col]
        test_df[col] = test_df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )
    test_df = test_df.dropna()
    if drop_duplicates:
        test_df = test_df.drop_duplicates(subset=attribute_columns)
    test_df = order_cols(test_df)

    print(' Length of testing data', len(test_df))
    test_df = order_cols(test_df)
    return test_df


def create_train_test_sets(
    train_files,
    test_files
):
    global use_cols
    global save_dir
    global column_value_filters
    global CONFIG
    global attribute_columns
    
    train_df_file = os.path.join(save_dir, 'train_data.csv')
    test_df_file = os.path.join(save_dir, 'test_data.csv')
    column_valuesId_file = 'column_valuesId_dict.pkl'
    column_valuesId_path = os.path.join( save_dir,column_valuesId_file)

    # --- Later on - remove using the saved file ---- #
    if os.path.exists(train_df_file) and os.path.exists(test_df_file) and False:
        train_df = pd.read_csv(train_df_file)
        test_df = pd.read_csv(test_df_file)

        with open(column_valuesId_path, 'rb') as fh:
            col_val2id_dict = pickle.load(fh)

        return train_df, test_df, col_val2id_dict

    train_df = clean_train_data(train_files)
    train_df = order_cols(train_df)
    
    for attr,_type in CONFIG['data_types'].items():
        if _type == 'int':
            train_df[attr] = train_df[attr].apply(_convert_to_int)
            train_df = train_df.dropna(subset=[attr])
    train_df = order_cols(train_df)
    train_df, col_val2id_dict = convert_to_ids(
        train_df,
        save_dir
    )

    # Create dataframes for each domain mapping
    for domain, _dict_ in col_val2id_dict.items():
        sqlite.get_engine()
        _dict_ = OrderedDict(_dict_)
        _df_1 = pd.DataFrame(
            {'entity_ID': list(_dict_.values()),
             domain : list(_dict_.keys())
            }
        )

        sqlite_conn = sqlite().get_engine()

        table_name = domain
        _df_2 = pd.read_sql("select * from {}".format(table_name), sqlite_conn)
        # Merge
        _df_ = _df_1.merge(
            _df_2,
            on = domain,
            how= 'inner'
        )
        _df_ = _df_.dropna()
        _df_['ID'] = _df_['ID'].astype(int)
        _df_ = _df_.rename(columns={'ID':'global_entity_ID', domain : 'value'})
        _df_.to_csv(
            os.path.join(save_dir, domain + '.csv'), index=False
        )

    train_df = train_df.drop_duplicates(subset=attribute_columns)
    print('Length of train data ', len(train_df))
    train_df = order_cols(train_df)
    '''
         test data preprocessing
    '''
    # combine test data into 1 file :
    list_test_df = [
        pd.read_csv(_file, low_memory=False, usecols=use_cols)
        for _file in test_files
    ]
    list_test_df = [_.dropna() for _ in list_test_df]
    list_test_df = HSCode_cleanup(list_test_df)

    test_df = None
    for _df in list_test_df:
        if test_df is None:
            test_df = _df
        else:
            test_df = test_df.append(_df)

    print('size of  Test set ', len(test_df))

    test_df = setup_testing_data(
        test_df,
        col_val2id_dict
    )

    test_df.to_csv(test_df_file, index=False)
    train_df.to_csv(train_df_file, index=False)
    # Save data_dimensions.csv ('column', dimension')
    dim_df = pd.DataFrame(columns=['column', 'dimension'])
    for col in attribute_columns:
        _count = len(col_val2id_dict[col])
        dim_df = dim_df.append({'column': col, 'dimension': _count},
                               ignore_index=True
                               )

    dim_df.to_csv(os.path.join(save_dir, 'data_dimensions.csv'), index=False)
    # -----------------------
    # Save col_val2id_dict
    # -----------------------
    with open(column_valuesId_path, 'wb') as fh:
        pickle.dump(col_val2id_dict, fh, pickle.HIGHEST_PROTOCOL)

    return train_df, test_df, col_val2id_dict


#  -------------------------------- #

