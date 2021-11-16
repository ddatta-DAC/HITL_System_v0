# Serve two kinds of sankey diagram

# ---------------------
# For each HSCode is the reference
# ShipmentOrigin, PortOfLading, HSCode, PortOfUnlading, ShipmentDestination
# ConsigneePanjivaID, HSCode, ShipperPanjivaID
# ---------------------
from redisStore import redisUtil
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
from plotly import io
from glob import glob
import multiprocessing as MP
from joblib import Parallel,delayed
from tqdm import tqdm 

sys.path.append('./..')
sys.path.append('./../..')
from DB_Ingestion.sqlite_engine import sqlite

SQL_Conn = sqlite.get_engine()
redis_obj = redisUtil.redisStore
html_saveDir = None

DATA_LOC = None
subDIR = None
json_saveDir = None
anomaly_result_dir = None
ID_col = 'PanjivaRecordID'
'''
#================================================
In both cases the anchoring element is HSCode

Diagram type 1: ShipmentOrigin', 'PortOfLading', 'HSCode', 'PortOfUnlading', 'ShipmentDestination
Diagram type 2: ConsigneeName, HSCode, ShipperName

# ================================================
'''

'''
Call this function with import 
_html_saveDir is a cache where the figures can be stored
'''

def initialize(
        _DATA_LOC=None,
        _subDIR=None,
        _html_saveDir=None,
        _json_saveDir =None,
        _anomaly_result_dir = None
):
    global redis_obj, DATA_LOC, subDIR, html_saveDir, json_saveDir,  anomaly_result_dir 
    anomaly_result_dir = _anomaly_result_dir
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    redis_obj.ingest_record_data(DATA_LOC, subDIR)
    if _html_saveDir is None:
        html_saveDir = './Cache_sankeyHTML'
    else:
        html_saveDir = _html_saveDir
    html_saveDir = html_saveDir + '_' + _subDIR
    Path(html_saveDir).mkdir(exist_ok=True, parents=True)
    if _json_saveDir is None:
        json_saveDir = './jsonCache'
    else:
        json_saveDir = _json_saveDir

    Path(json_saveDir).mkdir(exist_ok=True, parents=True)
    return

'''
Preload function to fetch and store results
'''
def __preload__(count=1000):
    global DATA_LOC, subDIR, anomaly_result_dir, ID_col
    cpu_count = MP.cpu_count()
    # fetch the record ids
    ad_result = pd.read_csv(
        os.path.join(anomaly_result_dir, subDIR, 'AD_output.csv'), index_col=None
    )
    samples = ad_result.rename(columns={'rank': 'score'})
    samples = samples.sort_values(by='score', ascending=True)
    count = min(len(samples) // 3, count)
    df = samples.head(count).copy(deep=True)
    del df['score']
    recordID_list = df[ID_col].astype(int).tolist()
    
#     for id in tqdm(recordID_list):
#         get_record_entityEmbeddings(int(id))
    
    Parallel(n_jobs=cpu_count, prefer="threads")(
        delayed(get_sankey_diagram)(int(id),1,) for id in tqdm(recordID_list)
    )
    Parallel(n_jobs=cpu_count, prefer="threads")(
        delayed(get_sankey_diagram)(int(id),2,) for id in tqdm(recordID_list)
    )
    return


def obtain_data(
        PanjivaRecordID,
        diagram_type=1,
        _debug=False
):
    global SQL_Conn
    global redis_obj
    global DATA_LOC
    global subDIR

    if diagram_type == 1:
        domain_subset = [
            'ShipmentOrigin', 'PortOfLading', 'HSCode', 'PortOfUnlading', 'ShipmentDestination'
        ]
    elif diagram_type == 2:
        domain_subset = [
            'ConsigneePanjivaID', 'HSCode', 'ShipperPanjivaID'
        ]

    record = redis_obj.fetch_data(
        str(int(PanjivaRecordID))
    )

    tmp = pd.read_csv(os.path.join(DATA_LOC, subDIR, 'HSCode.csv'), index_col=None)
    hscode_value = tmp.loc[tmp['entity_ID'] == record['HSCode']]['value'].values[0]
    if diagram_type == 1:
        _query_str = 'select {} from Records where HSCode={}'.format(','.join(domain_subset), hscode_value)

    elif diagram_type == 2:
        _query_str = 'select ConsigneeName, HSCode, ShipperName from Records ' \
                     'JOIN ConsigneePanjivaID  on  ConsigneePanjivaID.ConsigneePanjivaID=Records.ConsigneePanjivaID ' \
                     'JOIN ShipperPanjivaID on ShipperPanjivaID.ShipperPanjivaID=Records.ShipperPanjivaID ' \
                     'where  HSCode={}'.format(hscode_value)
    if _debug:
        print(_query_str)
    # ------------------------------
    # Use the sqllite  DB to read in all records that match HSCode
    # ------------------------------
    df = pd.read_sql(
        _query_str,
        con=SQL_Conn,
        index_col=None
    )
    if _debug:
        print('Data fetched ', len(df))

    return df


'''
return_type:
1 : Return plotly figure
2 : Return html [preferred]
3 : Write to HTML and return the file name
 
link_count_upper_bound 
if set to None 
    shows all link, 
else 
    limits to top <link_count_upper_bound> links - prevents clutter 
'''


def get_sankey_diagram(
        PanjivaRecordID,
        diagram_type=1,
        link_count_upper_bound=100,
        return_type=2,
        fig_height=600,
        use_cache=True
):
    global DATA_LOC
    global subDIR
    global html_saveDir
    global json_saveDir

    fig = None
    fig_json_fname = 'Sankey_{}_{}'.format(PanjivaRecordID, diagram_type) + '.json'
    fig_json_path = os.path.join(json_saveDir, fig_json_fname)
    
    fname = 'Sankey_{}_{}'.format(PanjivaRecordID, diagram_type) + '.html'
    fpath = os.path.join(html_saveDir, fname)
    
    if use_cache and os.path.exists(fpath) and return_type == 2 :
        fh = open(fpath, 'r')
        html_content = fh.read()
        fh.close()
        return html_content
    
    if use_cache and os.path.exists(fig_json_path) :
        with open(fig_json_path, 'r') as f:
            fig = io.from_json(f.read())

    # fetch dat if figure does not already exists
    if fig is None:
        # -----------------------
        # Read data
        # -----------------------
        df = obtain_data(
            PanjivaRecordID,
            diagram_type=diagram_type
        )

        if diagram_type == 1:
            domain_subset = [
                'ShipmentOrigin', 'PortOfLading', 'HSCode', 'PortOfUnlading', 'ShipmentDestination'
            ]
        elif diagram_type == 2:
            domain_subset = [
                'ConsigneeName', 'HSCode', 'ShipperName'
            ]

        # convert all the names to labels
        all_vals = []
        for col in list(df.columns):
            df[col] = df[col].apply(lambda x: col + ':' + x)
            vals = set(df[col].values)
            all_vals.extend(vals)

        label2id = {e[1]: e[0] for e in enumerate(all_vals, 0)}
        # replace in df
        df_1 = df.copy(deep=True)
        for col in list(df_1.columns):
            df_1[col] = df_1[col].parallel_apply(lambda x: label2id[x])

        source_list = []
        target_list = []
        value_list = []

        # --------------------------------------
        # Generate links between entities
        # --------------------------------------
        for a, b in zip(domain_subset[:-1], domain_subset[1:]):
            _df_ = df_1[[a, b]].groupby([a, b]).size().reset_index(name='count')
            if link_count_upper_bound is not None:
                _df_ = _df_.sort_values(by=['count'], ascending=False).head(link_count_upper_bound)
            source_list.extend(_df_[a].values.tolist())
            target_list.extend(_df_[b].values.tolist())
            value_list.extend(_df_['count'].values.tolist())
            print(len(target_list), len(source_list), len(value_list))

        if diagram_type == 1:
            title = 'Sankey Diagram Type 1'
        elif diagram_type == 2:
            title = 'Sankey Diagram Type 2'

        # ---------------use_cache--
        # create a graph for sankey diagram
        # -----------------
        label_list = list(all_vals)
        label_list = [_.split(':') for _ in label_list]
        link_source = source_list
        link_target = target_list

        fig = go.Figure(
            data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label_list
                ),
                link=dict(
                    source=link_source,
                    target=link_target,
                    value=value_list
                ))])

        fig.update_layout(
            font_size=8,
            paper_bgcolor='white'
        )
        
        fig.update_layout(
            autosize=True,
            font=dict(
                family="Arial",
                size=12,
                color="Black"
            ),
            plot_bgcolor='rgba(250,250,250,0.15)'
        )
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig.update_layout(height=fig_height)

        if use_cache:
            fig_json = io.to_json(fig)
            with open(fig_json_path, 'w') as f:
                f.write(fig_json)

    if return_type == 1:
        return fig
    elif return_type == 2:
        fname = 'Sankey_{}_{}'.format(PanjivaRecordID, diagram_type) + '.html'
        fpath = os.path.join(html_saveDir, fname)
        fig.write_html(
            fpath, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False
        )
        
        html_String = io.to_html(fig, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        return html_String
    elif return_type == 3:
        fname = 'Sankey_{}_{}'.format(PanjivaRecordID, diagram_type) + '.html'
        fpath = os.path.join(html_saveDir, fname)
        fig.write_html(
            fpath, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False
        )
        return fpath

    
# ----------------------------------------
'''

jsonCacheDir = './jsonCache'
htmlCacheDir = './htmlCache'

sankey.initialize(
    _DATA_LOC='./../generated_data_v1/us_import',
    _subDIR='01_2016',
    _html_saveDir=htmlCacheDir,
    _json_saveDir=jsonCacheDir,
)

# [NOTE] :: needs to have 2 calls
# Place 2 diagrams in 2 tabs.
# Call with type diagram_type= 1
# Visualization heading should be   'ShipmentOrigin - PortOfLading - HSCode - PortOfUnlading - ShipmentDestination'
get_sankey_diagram(
        PanjivaRecordID,
        diagram_type=1,
        link_count_upper_bound=100,
        return_type=2,
        fig_height=600,
        use_cache=True
)
# Call with type diagram_type=2
# Visualization heading should be  'ConsigneeName - HSCode - ShipperName'
get_sankey_diagram(
        PanjivaRecordID,
        diagram_type=2,
        link_count_upper_bound=100,
        return_type=2,
        fig_height=600,
        use_cache=True
)

'''