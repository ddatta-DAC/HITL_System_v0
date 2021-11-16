#!/usr/bin/env python
# coding: utf-8
import networkx as nx
import os
from pandarallel import pandarallel

pandarallel.initialize()
import numpy as np
from scipy.spatial.distance import cosine
# import matplotlib.pyplot as plt
from pyvis.network import Network
from sqlalchemy import create_engine
import pandas as pd
import faiss
from tqdm import tqdm
from IPython.display import HTML
from joblib import Parallel, delayed
import multiprocessing as MP
import sys
from pathlib import Path

sys.path.append('./../')
sys.path.append('./../..')
from DB_Ingestion.sqlite_engine import sqlite
from redisStore import redisUtil
import pickle

# ========================================================================================================
SQL_conn = sqlite().get_engine()
DATA_LOC = None
subDIR = None
html_saveDir = None
json_saveDir = None
redis_obj = redisUtil.redisStore
column_values2id = None
DATA_LOC = None
subDIR = None
html_cache = None
DATA_LOC = None
saved_emb_loc = None
subDIR = None
NN_count = 25
df_cache = None
anomaly_result_dir = None
ID_col = 'PanjivaRecordID'


# SQL_conn = create_engine('sqlite:///./../../DB/wwf.db', echo=False)

# =========================================================================================================
def preprocess_data(
        num_NN=10
):
    global DATA_LOC, saved_emb_loc, subDIR, df_cache
    consignee_embedding = np.load(
        os.path.join(saved_emb_loc, subDIR, '{}_gnn_{}.npy'.format('ConsigneePanjivaID', 64))).astype(np.float32)
    shipper_embedding = np.load(
        os.path.join(saved_emb_loc, subDIR, '{}_gnn_{}.npy'.format('ShipperPanjivaID', 64))).astype(np.float32)

    vec_shipper = shipper_embedding
    vec_consignee = consignee_embedding

    tags_consignee = ['Consignee_{}'.format(_) for _ in np.arange(consignee_embedding.shape[0]).astype(int)]
    tags_shipper = ['Shipper_{}'.format(_) for _ in np.arange(shipper_embedding.shape[0]).astype(int)]

    # Create a local entity to serial id
    serial_ids = np.arange(consignee_embedding.shape[0] + shipper_embedding.shape[0]).astype(int)
    entity_id_list = np.arange(consignee_embedding.shape[0]).astype(int).tolist() + np.arange(
        shipper_embedding.shape[0]).astype(int).tolist()
    serialID2entityID = {k: v for k, v in zip(serial_ids, entity_id_list)}
    entityID2serialID = {v: k for k, v in zip(serial_ids, entity_id_list)}

    consignee_serial_ids = np.arange(consignee_embedding.shape[0]).astype(int)
    shipper_serial_ids = np.arange(shipper_embedding.shape[0]).astype(int)

    # ===================
    # Create FAISS object
    # ===================
    vectors = np.vstack([consignee_embedding, shipper_embedding]).astype(np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype(np.float32))

    def calc_sim(row):
        t = row['ConsigneePanjivaID']
        s = row['ShipperPanjivaID']
        _sim = 1 - cosine(vec_shipper[s], vec_consignee[t])
        return _sim

    # ----------------------
    # 1. Read in main data
    # 2. Create edges
    # 3. Use k-NN to add in supplementary edges ( TODO)
    # ----------------------

    train_df = pd.read_csv(os.path.join(DATA_LOC, subDIR, 'train_data.csv'), index_col=None)
    df = train_df[['ConsigneePanjivaID', 'ShipperPanjivaID']].drop_duplicates()

    # Find the nearest neighbor for all consignee
    pairs_1 = find_NearestNbrs(
        index_obj=index,
        serialID2entityID=serialID2entityID,
        vector=consignee_embedding,
        id_list1=consignee_serial_ids,
        id_list2=shipper_serial_ids,
        num_NN=num_NN
    )

    # Find the nearest neighbor for all shippers
    pairs_2 = find_NearestNbrs(
        index_obj=index,
        serialID2entityID=serialID2entityID,
        vector=shipper_embedding,
        id_list1=shipper_serial_ids,
        id_list2=consignee_serial_ids,
        num_NN=num_NN
    )

    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)

    tmpdf_1 = pd.DataFrame({
        'ConsigneePanjivaID': pairs_1[:, 0],
        'ShipperPanjivaID': pairs_1[:, 1]
    })

    tmpdf_2 = pd.DataFrame({
        'ShipperPanjivaID': pairs_2[:, 0],
        'ConsigneePanjivaID': pairs_2[:, 1]
    })

    df_1 = tmpdf_1.copy()
    df_1 = df_1.append(tmpdf_2, ignore_index=True)
    df_1 = df_1.drop_duplicates()

    df_1['weight'] = df_1.apply(calc_sim, axis=1).reset_index(drop=True)
    # Normalize the wights
    _min = np.min(df_1['weight'])
    _max = np.max(df_1['weight'])
    df_1['weight'] = df_1['weight'].apply(lambda x: (x - _min) / (_max - _min))
    df_projected = df_1.copy(deep=True)

    # ------------------------------------
    # Add in the actual links
    # ------------------------------------
    df_actual = train_df.groupby(['ConsigneePanjivaID', 'ShipperPanjivaID']).size().reset_index(name='weight')
    df_actual['ConsigneePanjivaID'] = df_actual['ConsigneePanjivaID'].apply(lambda x: 'ConsigneePanjivaID-{}'.format(x))
    df_actual['ShipperPanjivaID'] = df_actual['ShipperPanjivaID'].apply(lambda x: 'ShipperPanjivaID-{}'.format(x))
    df_projected['ConsigneePanjivaID'] = df_projected['ConsigneePanjivaID'].apply(
        lambda x: 'ConsigneePanjivaID-{}'.format(x))
    df_projected['ShipperPanjivaID'] = df_projected['ShipperPanjivaID'].apply(lambda x: 'ShipperPanjivaID-{}'.format(x))

    fname = os.path.join(df_cache, 'df_actual.csv')
    df_actual.to_csv(fname, index=None)
    fname = os.path.join(df_cache, 'df_projected.csv')
    df_projected.to_csv(fname, index=None)
    return


# ============================================
# Find the nearest neigbhbors using Embedding
# The return is a list of pair of entity_ids
# ============================================
def find_NearestNbrs(
        index_obj,
        serialID2entityID,
        vector,
        id_list1,
        id_list2,
        num_NN
):
    print('Finding nearest neighbors...')
    distances, NN_ids = index_obj.search(vector, k=num_NN * 4)
    # ---------------------------------------------------
    # Filter nbr to be of other type (bipartite graph)
    # ---------------------------------------------------

    # vectors are ordered internally (intra-domain)
    entity_ids = [serialID2entityID[i] for i in id_list1]

    def aux_check(_id_, nn_list, validation_list, num_NN):
        filtered_nn = [_nbr for _nbr in nn_list if _nbr in validation_list][:num_NN]
        return (_id_, filtered_nn)

    validation_list = id_list2
    res = Parallel(n_jobs=MP.cpu_count())(
        delayed(aux_check)(_id_, _nn_, validation_list, num_NN) for _id_, _nn_ in zip(entity_ids, NN_ids)
    )

    pairs = []
    for pair in res:
        for _item in pair[1]:
            pairs.append((pair[0], serialID2entityID[_item]))
    print(len(pairs))
    return pairs


def initialize(
        _DATA_LOC,
        _subDIR,
        _saved_emb_loc,
        _df_cache,
        _html_cache,
        db_loc,
        _anomaly_result_dir
):
    global DATA_LOC, subDIR, anomaly_result_dir, redis_obj, column_values2id, NN_count, df_cache, html_cache, saved_emb_loc, SQL_conn
    anomaly_result_dir = _anomaly_result_dir
    
    print(os.path.exists(db_loc) , os.getcwd())
    SQL_conn = create_engine(
        'sqlite:///{}'.format(db_loc),
        echo=False
    )
    print(SQL_conn)
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    saved_emb_loc = _saved_emb_loc

    with open(os.path.join(DATA_LOC, subDIR, 'col_val2id_dict.pkl'), 'rb') as fh:
        column_values2id = pickle.load(fh)
    df_cache = _df_cache
    html_cache = _html_cache
    Path(df_cache).mkdir(exist_ok=True, parents=True)
    Path(html_cache).mkdir(exist_ok=True, parents=True)
    redis_obj.ingest_record_data(DATA_LOC, subDIR)
    preprocess_data(10)
    return


def obtain_node_display_data(
        node_type,
        _id
):
    global SQL_conn
    global DATA_LOC, subDIR
    global column_values2id

    # get the actual value from the id passed
    __ID__ = [value for value, e_id in column_values2id[node_type].items() if e_id == int(_id)][0]
    display_str = ''

    if node_type == 'ConsigneePanjivaID':
        df = pd.read_sql(
            "select {},{},{},{} from {} where {}={}".format(
                'ConsigneePanjivaID', 'ConsigneeName', 'ConsigneeCity', 'ConsigneeCountry', 'ConsigneePanjivaID',
                'ConsigneePanjivaID', __ID__
            ),
            SQL_conn,
            index_col=None
        )
        display_str = 'Consignee :: '
        display_str += '<br>'.join(
            [str(df['ConsigneeName'].values[0]), str(df['ConsigneeCity'].values[0]),
             str(df['ConsigneeCountry'].values[0])]
        )

        df = pd.read_sql('select count(*) as count  from Records where ConsigneePanjivaID={}'.format(__ID__), SQL_conn,
                         index_col=None)
        count = df['count'].values[0]

    if node_type == 'ShipperPanjivaID':
        df = pd.read_sql(
            "select {},{},{},{} from {} where {}={}".format(
                'ShipperPanjivaID', 'ShipperName', 'ShipperCity', 'ShipperCountry', 'ShipperPanjivaID',
                'ShipperPanjivaID', __ID__
            ),
            SQL_conn,
            index_col=None
        )
        display_str = 'Shipper :: '
        display_str += '<br>'.join(
            [str(df['ShipperName'].values[0]), str(df['ShipperCity'].values[0]), str(df['ShipperCountry'].values[0])]
        )
        df = pd.read_sql('select count(*) as count from Records where ShipperPanjivaID={}'.format(__ID__), SQL_conn,
                         index_col=None)
        count = df['count'].values[0]
    count = int(10 + (np.log10(count) + 1) * 2)
    return __ID__, display_str, count


'''
Function to preload everything into a cache
In this case on disk : html_cache
The idea is to precalculate viz html for top-K records in each epoch 
'''


def __preload__(count=2500):
    from onlineUpdateModule import data_handler
    global DATA_LOC, subDIR, anomaly_result_dir, ID_col
    cpu_count = MP.cpu_count()
    # fetch the record ids
    print(anomaly_result_dir, subDIR)
    ad_result = pd.read_csv(
        os.path.join(anomaly_result_dir, subDIR, 'AD_output.csv'), index_col=None
    )
    samples = ad_result.rename(columns={'rank': 'score'})
    samples = samples.sort_values(by='score', ascending=True)
    count = min(len(samples) // 3, count)
    df = samples.head(count).copy(deep=True)
    del df['score']
    recordID_list = df[ID_col].tolist()
    
    for id in tqdm(recordID_list):
        visualize(int(id))
    Parallel(n_jobs=cpu_count)(
        delayed(visualize)(int(id)) for id in tqdm(recordID_list)
    )
    return df


# ================================================================
# Main function
# ================================================================
def visualize(
        PanjivaRecordID,
        fig_height='720px',
        fig_width='100%',
        title=False,
        return_type=2
):
    global DATA_LOC, subDIR, redis_obj, df_cache, html_cache
    signature_fname = 'consigneeShipper_{}_{}_{}.html'.format(PanjivaRecordID, fig_height, fig_width)

    f_path = os.path.join(html_cache,signature_fname)

    if os.path.exists(f_path) and return_type == 2:
        return os.path.join(html_cache, signature_fname)

    # -----------------------
    # Read in saved dataframes 
    df_actual = pd.read_csv(os.path.join(df_cache, 'df_actual.csv'), index_col=None)

    df_0 = pd.read_csv(os.path.join(df_cache, 'df_actual.csv'), index_col=None)
    df_1 = pd.read_csv(os.path.join(df_cache, 'df_projected.csv'), index_col=None)

    # -----------------------
    record = redis_obj.fetch_data(str(PanjivaRecordID))
    
    consignee = 'ConsigneePanjivaID-' + str(record['ConsigneePanjivaID'])
    shipper = 'ShipperPanjivaID-' + str(record['ShipperPanjivaID'])

    if consignee is not None and shipper is not None:
        df_0 = df_0.loc[(df_0['ConsigneePanjivaID'] == consignee) | (df_0['ShipperPanjivaID'] == shipper)]

    df_pred_consignee = df_1.loc[(df_1['ShipperPanjivaID'] == shipper)].sort_values(by='weight', ascending=False).head(
        10)
    df_pred_shipper = df_1.loc[(df_1['ConsigneePanjivaID'] == consignee)].sort_values(by='weight',
                                                                                      ascending=False).head(10)

    if title is True:
        title_text = 'Network of Consignee & Shippers'
    else:
        title_text = ''
    # ---------------------------

    # Add in secondary edges
    df_2 = df_actual.loc[
        (df_actual['ConsigneePanjivaID'].isin(df_pred_consignee['ConsigneePanjivaID'])) |
        (df_actual['ShipperPanjivaID'].isin(df_pred_shipper['ShipperPanjivaID']))
        ]
    # ---------------------------
    # Create a networkx graph 
    nx_graph = nx.Graph()
    # Add nodes
    consignee_nodes = df_0['ConsigneePanjivaID'].values.tolist()
    consignee_nodes += df_pred_consignee['ConsigneePanjivaID'].values.tolist()
    consignee_nodes += df_pred_shipper['ConsigneePanjivaID'].values.tolist()
    consignee_nodes += df_2['ConsigneePanjivaID'].values.tolist()
    consignee_nodes = list(set(consignee_nodes))

    shipper_nodes = df_0['ShipperPanjivaID'].values.tolist()
    shipper_nodes += df_pred_consignee['ShipperPanjivaID'].values.tolist()
    shipper_nodes += df_pred_shipper['ShipperPanjivaID'].values.tolist()
    shipper_nodes += df_2['ShipperPanjivaID'].values.tolist()
    shipper_nodes = list(set(shipper_nodes))

    # Obtain node data 
    def aux_get_data(node_id):
        node_type, _id = node_id.split('-')
        _ID_, node_descriptor, node_size = obtain_node_display_data(node_type, _id)
        return (node_id, node_descriptor, _ID_, node_size)

    node_data_consignee = Parallel(n_jobs=MP.cpu_count(), prefer="threads")(
        delayed(aux_get_data)(node, ) for node in consignee_nodes
    )
    node_data_shipper = Parallel(n_jobs=MP.cpu_count(), prefer="threads")(
        delayed(aux_get_data)(node, ) for node in shipper_nodes
    )
    for cn in node_data_consignee:
        nx_graph.add_node(cn[0], label=cn[2], title=cn[1], size=cn[3], color='Coral')

    for cn in node_data_shipper:
        nx_graph.add_node(cn[0], label=cn[2], title=cn[1], size=cn[3], color='MediumSlateBlue')

    for _type, _df in zip(['actual', 'actual', 'predicted', 'predicted'],
                          [df_0, df_2, df_pred_consignee, df_pred_shipper]):
        sources = _df['ConsigneePanjivaID']
        targets = _df['ShipperPanjivaID']
        weights = _df['weight']
        edge_data = zip(sources, targets, weights)
        for e in edge_data:
            src = e[0]
            dst = e[1]
            w = e[2]
            if _type == 'predicted' and (src == shipper or dst == consignee or dst == shipper or src == consignee):
                nx_graph.add_edge(src, dst, weight=w, color='red')
            else:
                nx_graph.add_edge(src, dst, weight=w, color='blue')

    largest_cc = max(nx.connected_components(nx_graph), key=len)
    G = nx_graph.subgraph(largest_cc).copy()

    net = Network(
        height=fig_height,
        width=fig_width,
        bgcolor="white",
        font_color="black",
        notebook=False,
        heading=title_text
    )

    net.from_nx(G)
    net.barnes_hut()
    net.set_options(
        """
        var options = {
          "nodes": {
            "borderWidthSelected": 4,
            "color": {
              "background": "rgba(163,252,11,1)"
            }
          }
        }
        """)
    f_path = os.path.join(html_cache, signature_fname)
    net.write_html(f_path)
    
    if return_type == 1:
        return net
    else:
        f = open(f_path, "r")
        _html_ = f.read()
        f.close()
        return _html_

# ==============================================================================
# initialize(
#     _DATA_LOC ='./../generated_data_v1/us_import',
#     _saved_emb_loc =  './../GNN/saved_model_gnn',
#     _subDIR = '01_2016',
#     _html_cache= 'networkViz_htmlCache',
#     _df_cache = 'networkViz_dfCache',
#     db_loc ='/../DB/wwf.db',
#     anomaly_result_dir = './../AD_model/combined_output'
# )
# __preload__()
# 
# html_path = visualize( 
#     PanjivaRecordID ='120888026',
#     fig_width='100%', 
#     title=False, 
#     fig_height='920px', 
#     return_type = 2
# )
# ==============================================================================
