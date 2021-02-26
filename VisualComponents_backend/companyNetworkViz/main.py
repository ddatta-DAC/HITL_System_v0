#!/usr/bin/env python
# coding: utf-8

import os
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
import faiss
from IPython.display import HTML
from joblib import Parallel,delayed
import multiprocessing as MP
import sys
from pathlib import Path
sys.path.append('./../')
sys.path.append('./../..')
from DB_Ingestion.sqlite_engine import sqlite
from redisStore import redisUtil
import pickle
from DB_Ingestion.sqlite_engine import sqlite

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


# =========================================================================================================


def preprocess_data (
    num_NN = 25
):
    global DATA_LOC, saved_emb_loc, subDIR, df_cache
    consignee_embedding = np.load( os.path.join(saved_emb_loc,subDIR,'{}_gnn_{}.npy'.format('ConsigneePanjivaID',64))).astype(np.float32)
    shipper_embedding = np.load(os.path.join(saved_emb_loc,subDIR, '{}_gnn_{}.npy'.format('ShipperPanjivaID',64))).astype(np.float32)

    vec_shipper = shipper_embedding
    vec_consignee = consignee_embedding
    tags_consignee = ['Consignee_{}'.format(_) for _ in np.arange(consignee_embedding.shape[0]).astype(int)]
    tags_shipper = ['Shipper_{}'.format(_) for _ in np.arange(shipper_embedding.shape[0]).astype(int)]

    # Create a local entity to serial id
    serial_ids = np.arange(consignee_embedding.shape[0] + shipper_embedding.shape[0] ).astype(int)
    entity_id_list = np.arange(consignee_embedding.shape[0]).astype(int).tolist()  + np.arange(shipper_embedding.shape[0]).astype(int).tolist()
    serialID2entityID = { k:v for k,v in zip(serial_ids , entity_id_list)}
    entityID2serialID = { v:k for k,v in zip(serial_ids , entity_id_list)}
    
    consignee_serial_ids = np.arange(consignee_embedding.shape[0]).astype(int)
    shipper_serial_ids = np.arange(shipper_embedding.shape[0]).astype(int)
    # ===================
    # Create FAISS object
    # ===================
    
    vectors  = np.vstack([consignee_embedding, shipper_embedding]).astype(np.float32)
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
    
    train_df = pd.read_csv( os.path.join(DATA_LOC, subDIR, 'train_data.csv'),index_col=None)
    df = train_df[['ConsigneePanjivaID','ShipperPanjivaID']].drop_duplicates()
    
    # Find the nearest neighbor for consignee
    pairs_1 = find_NearestNbrs (
        index_obj = index, 
        serialID2entityID = serialID2entityID,
        vector = consignee_embedding, 
        id_list1 = consignee_serial_ids, 
        id_list2 = shipper_serial_ids,
        num_NN = num_NN
    )
    
    # Find the nearest neighbor for consignee
    pairs_2 = find_NearestNbrs (
        index_obj = index,
        serialID2entityID = serialID2entityID,
        vector = shipper_embedding, 
        id_list1 = shipper_serial_ids,
        id_list2 = consignee_serial_ids,
        num_NN = num_NN
    )

    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)
                           
    
    tmpdf_1 = pd.DataFrame( {
        'ConsigneePanjivaID': pairs_1[:,0],
        'ShipperPanjivaID': pairs_1[:,1]
    })
    
    tmpdf_2 = pd.DataFrame( {
        'ShipperPanjivaID': pairs_2[:,0],
        'ConsigneePanjivaID': pairs_2[:,1]
    })
    
    df_1 = tmpdf_1.copy()
    df_1 = df_1.append(tmpdf_2,ignore_index=True)
    df_1 = df_1.drop_duplicates()
    
    df_1['weight'] = df_1.apply(calc_sim, axis=1).reset_index(drop=True)
    df_projected =  df_1.copy(deep=True)
    # ------------------------------------
    # Add in the actual links
    # ------------------------------------
    df_actual = train_df.groupby(['ConsigneePanjivaID','ShipperPanjivaID']).size().reset_index(name='weight')
    df_actual['ConsigneePanjivaID'] = df_actual['ConsigneePanjivaID'].apply(lambda x: 'ConsigneePanjivaID-{}'.format(x))
    df_actual['ShipperPanjivaID'] = df_actual['ShipperPanjivaID'].apply(lambda x: 'ShipperPanjivaID-{}'.format(x)) 
    df_projected['ConsigneePanjivaID'] = df_projected['ConsigneePanjivaID'].apply(lambda x: 'ConsigneePanjivaID-{}'.format(x))
    df_projected['ShipperPanjivaID'] = df_projected['ShipperPanjivaID'].apply(lambda x: 'ShipperPanjivaID-{}'.format(x)) 
   
    fname = os.path.join(df_cache,'df_actual.csv')
    df_actual.to_csv(fname, index=None)
    fname = os.path.join(df_cache,'df_projected.csv')
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
    distances, NN_ids = index_obj.search( vector, k = num_NN*4) 
    
    # ---------------------------------------------------
    # Filter nbr to be of other type (bipartite graph)
    # ---------------------------------------------------
    
    # vectors are ordered internally (intra-domain)
    entity_ids = [ serialID2entityID[i] for i in id_list1 ]
    
    def aux_check(_id_, nn_list, validation_list, num_NN):
        filtered_nn = [ _nbr for _nbr in nn_list if _nbr in validation_list][:num_NN]
        return ( _id_, filtered_nn)
    
    validation_list = id_list2
    res = Parallel(n_jobs=MP.cpu_count()) (
        delayed(aux_check)( _id_ , _nn_, validation_list, num_NN ) for _id_, _nn_ in zip(entity_ids , NN_ids)
    )
    
    pairs = []
    for pair in res:
        for _item in pair[1]:
            pairs.append((pair[0], serialID2entityID[_item]))
    
    return pairs


def initialize(
    _DATA_LOC,
    _subDIR,
    _saved_emb_loc,
    _df_cache,
    _html_cache
):
    global DATA_LOC, subDIR, redis_obj, column_values2id, NN_count, df_cache, html_cache, saved_emb_loc
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    saved_emb_loc = _saved_emb_loc
   
    with open(os.path.join(DATA_LOC, subDIR,'col_val2id_dict.pkl'), 'rb') as fh:
        column_values2id = pickle.load(fh)
    df_cache  = _df_cache
    html_cache = _html_cache
    Path(df_cache).mkdir(exist_ok=True,parents=True)
    Path(html_cache).mkdir(exist_ok=True,parents=True)
    redis_obj.ingest_record_data(DATA_LOC, subDIR)
    preprocess_data(NN_count)
    return 
    
    

def obtain_node_display_data(node_type, _id):
    global SQL_conn
    global DATA_LOC, subDIR
    global column_values2id
    
    # get the actual value from the id passed
    __ID__ = [value  for value, e_id  in column_values2id[node_type].items() if e_id == int(_id)][0]
    display_str = ''
    if node_type == 'ConsigneePanjivaID':
        df = pd.read_sql(
                "select {},{},{},{} from {} where {}={}".format(
                    'ConsigneePanjivaID', 'ConsigneeName', 'ConsigneeCity', 'ConsigneeCountry', 'ConsigneePanjivaID', 'ConsigneePanjivaID', __ID__
                ), 
                SQL_conn,
                index_col=None
        )
        display_str = 'Consignee :: '
        display_str += '<br>' .join(
                [str(df['ConsigneeName'].values[0]),str(df['ConsigneeCity'].values[0]),str(df['ConsigneeCountry'].values[0])]
        )
    if node_type == 'ShipperPanjivaID':
        df = pd.read_sql(
                "select {},{},{},{} from {} where {}={}".format(
                    'ShipperPanjivaID', 'ShipperName', 'ShipperCity', 'ShipperCountry', 'ShipperPanjivaID', 'ShipperPanjivaID', __ID__
                ), 
                SQL_conn,
                index_col=None
        )
        display_str = 'Shipper :: '
        display_str += '<br>' .join(
            [str(df['ShipperName'].values[0]), str(df['ShipperCity'].values[0]), str(df['ShipperCountry'].values[0])]
        )
    return __ID__, display_str


def visualize(
    PanjivaRecordID,
    fig_height = '720px',
    fig_width = '100%',
    title=False
):
    global DATA_LOC, subDIR, redis_obj, df_cache, html_cache
    signature_fname = 'consigneeShipper_{}_{}_{}.html'.format(PanjivaRecordID,fig_height,fig_width) 
    
    if os.path.exists(os.path.join(html_cache, signature_fname)):
        return os.path.join(html_cache, signature_fname)
    
    # -----------------------
    # Read in saved dataframes 
    df_0 = pd.read_csv(os.path.join(df_cache, 'df_actual.csv'),index_col=None)
    df_1 = pd.read_csv(os.path.join(df_cache, 'df_projected.csv'),index_col=None)

    
    record = redis_obj.fetch_data(str(PanjivaRecordID))
    consignee = 'ConsigneePanjivaID-' + str(record['ConsigneePanjivaID'])
    shipper = 'ShipperPanjivaID-' + str(record['ShipperPanjivaID'])
    
    if consignee is not None and  shipper is not None:
        df_0 = df_0.loc[(df_0['ConsigneePanjivaID']==consignee)|(df_0['ShipperPanjivaID']==shipper)]
        df_1 = df_1.loc[(df_1['ConsigneePanjivaID']==consignee)|(df_1['ShipperPanjivaID']==shipper)]
    elif consignee is not None:
        df_0 = graph_data.loc[df_0['ConsigneePanjivaID']==consignee]
        df_1 = graph_data.loc[df_1['ConsigneePanjivaID']==consignee]
    elif shipper is not None:
        df_0 = df_0.loc[df_0['ShipperPanjivaID']==shipper]
        df_1 = df_1.loc[df_1['ShipperPanjivaID']==shipper]
    
    if title is True:
        title_text = 'Network of Consignee & Shippers'
    else:
        title_text = ''
    _net = Network(
        height=fig_height, 
        width=fig_width,
        bgcolor="white", 
        font_color="black", 
        heading= title_text,
        notebook=False
    )
    
    # set the physics layout of the network
    _net.barnes_hut() 
    
    i = 0
    for _type, _df in zip(['actual','projected'],[df_0, df_1]):
        sources = _df['ConsigneePanjivaID']
        targets = _df['ShipperPanjivaID']
        weights = _df['weight']
        if _type =='actual':
            edge_arrow_type = False
        else:
            edge_arrow_type = True
        edge_data = zip(sources, targets, weights)
        for e in edge_data:
            src = e[0]
            dst = e[1]
            w = e[2]
            _net.add_node(src, src, title=src, group=2*i+0,labelHighlightBold = True,)
            _net.add_node(dst, dst, title=dst, group=2*i+1,labelHighlightBold = True,)
            _net.add_edge(src, dst, value=w, title = _type, arrowStrikethrough=edge_arrow_type)
            neighbor_map = _net.get_adj_list()
            
            # add neighbor data to node hover data
            for node in _net.nodes:
                node["value"] = len(neighbor_map[node['id']])*25
        i+=1
        _net.add_edge(consignee, shipper, value=1, title = 'actual', arrowStrikethrough=True)
    # ---------------------------------------------------
    # Set up the display data for each node
    # ---------------------------------------------------
    
    def aux(node_id):
        node_type, _id = node_id.split('-')
        _ID_, node_descriptor = obtain_node_display_data(node_type, _id)
        return (node_id,node_descriptor)
        
    node_display_str = Parallel(n_jobs=MP.cpu_count(),prefer="threads")(
        delayed(aux)(node['id'],) for node in _net.nodes
    )
    node_display_str = {i[0]:i[1] for i in node_display_str}
    for node in _net.nodes:
        node_id = node['id']   
        node['title'] = node_display_str[node_id]
    
#     _net.set_options('''
#     var options = {
#         "physics": {
#             "barnesHut": {
#               "gravitationalConstant": -10000,
#               "centralGravity": 0.15,
#               "springLength": 750,
#               "springConstant": 0.10,
#               "damping": 0.0,
#               "avoidOverlap": 0.6
#             },
#             "maxVelocity": 1,
#             "minVelocity": 0,
#             "timestep": 10
#           }
#     }
#     '''
#     )
    f_path = os.path.join(html_cache, signature_fname)
    _net.write_html(f_path)
    return f_path

# ==============================================================================
# initialize(
#     _DATA_LOC ='./../generated_data_v1/us_import',
#     _saved_emb_loc =  './saved_model_gnn',
#     _subDIR = '01_2016',
#     _html_cache= 'htmlCache',
#     _df_cache = 'dfCache'
# )


# html_path = visualize( PanjivaRecordID ='120901356',fig_width='720px', title=False)
# html_path = visualize( PanjivaRecordID ='121852671')
# ==============================================================================




