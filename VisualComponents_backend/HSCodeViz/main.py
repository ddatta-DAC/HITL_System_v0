import pandas as pd

import plotly.express as px
import pandas as pd
import sys
import os
from tqdm import tqdm
import plotly
import plotly.graph_objects as go
import plotly.io as pio
sys.path.append('./../')
sys.path.append('./../..')
import pickle
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing as MP
from DB_Ingestion.sqlite_engine import sqlite
SQL_conn =None
DATA_LOC = None
subDIR = None
ID_col = 'PanjivaRecordID'
anomaly_result_dir = None

def initialize(
    _DATA_LOC,
    _subDIR,
    _htmlSaveDir = None,
    _anomaly_result_dir = None
):
    global DATA_LOC, subDIR
    global SQL_conn
    global subDIR
    global htmlSaveDir
    global anomaly_result_dir
    subDIR = _subDIR
    anomaly_result_dir = _anomaly_result_dir 
    if _htmlSaveDir is None:
        htmlSaveDir = './htmlCache'
    else:
        htmlSaveDir = _htmlSaveDir
    htmlSaveDir = htmlSaveDir + '_' + str(_subDIR)
    Path(htmlSaveDir).mkdir(exist_ok=True,parents=True )
    
    SQL_conn = sqlite().get_engine()
    DATA_LOC = _DATA_LOC
    
    return 

'''
Function to preload everything into a cache
In this case on disk : html_cache
The idea is to precalculate viz html for top-K records in each epoch 
'''

def __preload__(count=1000):
    
    global DATA_LOC, subDIR, anomaly_result_dir, ID_col
    cpu_count = MP.cpu_count()
    # fetch the record ids
    ad_result = pd.read_csv(
        os.path.join(anomaly_result_dir, subDIR, 'AD_output.csv'), index_col=None
    )
    samples = ad_result.rename(columns={'rank': 'score'})
    samples = samples.sort_values(by='score', ascending=False)
    count = min(len(samples) // 3, count)
    df = samples.head(count).copy(deep=True)
    del df['score']
    recordID_list = df[ID_col].tolist()
    
#     for id in tqdm(recordID_list):
#         get_stackedComparisonPlots(int(id))
    Parallel(n_jobs=cpu_count, prefer="threads")(
        delayed(get_HSCode_distribution)(int(_id)) for _id in tqdm(recordID_list)
    )
    return 


# ========================================
# Return Radial sunburst chart for the HSCode breakdown of Consignee and Shipper Respectively
# In viz set header for Panel as 'HS Code distribution for Consignee' 'HS Code distribution for Shipper'
# ========================================
def get_HSCode_distribution(
    record_id, 
    return_type=2
):
    
    global subDIR, htmlSaveDir
    
    signature_1 = '{}_Consignee_hscode'.format(record_id)
    signature_2 = '{}_Shipper_hscode'.format(record_id)
    fpath_1 = os.path.join(htmlSaveDir, signature_1 + '.html')
    fpath_2 = os.path.join(htmlSaveDir, signature_2 + '.html')
    
    if os.path.exists(fpath_1) and os.path.exists(fpath_2):
        fh = open(fpath_1,'r')
        fig_html_c =  fh.read()
        fh.close()
        fh = open(fpath_2,'r')
        fig_html_s =  fh.read()
        fh.close()
        return fig_html_c, fig_html_s
        
    df = pd.read_sql(
                "select {},{} from Records where {}={}".format(
                    'ConsigneePanjivaID', 'ShipperPanjivaID', 'PanjivaRecordID', record_id
                ), 
                SQL_conn,
                index_col=None
        )

    _shipper = int(df['ShipperPanjivaID'])
    _consignee = int(df['ConsigneePanjivaID'])

    df_consignee = pd.read_sql(
                "select {},{} from Records where {}={}".format(
                    'ConsigneePanjivaID', 'HSCode', 'ConsigneePanjivaID', _consignee
                ), 
                SQL_conn,
                index_col=None
        )
    df_consignee = df_consignee.groupby(['HSCode']).size().reset_index(name='count')


    df_shipper = pd.read_sql(
                "select {},{} from Records where {}={}".format(
                    'ShipperPanjivaID', 'HSCode', 'ShipperPanjivaID', _shipper
                ), 
                SQL_conn,
                index_col=None
        )
    df_shipper = df_shipper.groupby(['HSCode']).size().reset_index(name='count')


    # create HSCode breakdowns
    df_shipper['HSCode_4'] = df_shipper['HSCode'].apply(lambda x: str(x[:4]))
    df_shipper['HSCode_2'] = df_shipper['HSCode'].apply(lambda x: str(x[:2]))
    df_consignee['HSCode_4'] = df_consignee['HSCode'].apply(lambda x: str(x[:4]))
    df_consignee['HSCode_2'] = df_consignee['HSCode'].apply(lambda x: str(x[:2]))

    fig_s = px.sunburst(df_shipper, path=['HSCode_2', 'HSCode_4', 'HSCode'], values='count')
    fig_s.update_layout(
        uniformtext=dict(minsize=16, mode='hide')
    )
    fig_s.update_layout(margin = dict(t=0, l=0, r=0, b=0))


    fig_c = px.sunburst(df_consignee, path=['HSCode_2', 'HSCode_4', 'HSCode'], values='count')
    fig_c.update_layout(
        uniformtext=dict(minsize=16, mode='hide')
    )
    fig_c.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    fig_html_c = pio.to_html(fig_c, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
    fig_html_s = pio.to_html(fig_s, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
    
    fig_c.write_html(fpath_1, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
    fig_s.write_html(fpath_2, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
    
    if return_type == 1: 
        return fig_c,fig_s
    elif return_type == 2:
        return fig_html_c, fig_html_s
    
# ---------------------------------

# initialize(
#     _DATA_LOC='./../generated_data_v1/us_import',
#     _subDIR='01_2016'
# )

# record_id = '121983256'
# record_id = '121092583'
