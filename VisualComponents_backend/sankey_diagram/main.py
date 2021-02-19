# Serve two kinds of sankey diagram

# ---------------------
# For each HSCode is the reference
# ShipmentOrigin, PortOfLading, HSCode, PortOfUnlading, ShipmentDestination
# ConsigneePanjivaID, HSCode, ShipperPanjivaID
# ---------------------
from redisStore import redisUtil
import os
import pandas as pd
import sys
sys.path.append('./..')
sys.path.append('./../..')
from DB_Ingestion.sqlite_engine import sqlite
import plotly.graph_objects as go
SQL_Conn = sqlite.get_engine()
redis_obj = redisUtil.redisStore

def initialize(
    DATA_LOC=None,
    subDIR=None
):
    global redis_obj
    redis_obj.ingest_record_data(DATA_LOC, subDIR)
    return 

def obtain_data(
        PanjivaRecordID,
        DATA_LOC=None,
        subDIR=None,
        diagram_type=1
):
    global SQL_Conn
    global redis_obj
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
    print( tmp.loc[tmp['entity_ID'] == record['HSCode']])
    
    hscode_value = tmp.loc[tmp['entity_ID'] == record['HSCode']]['value'].values[0]
    
    print('HSCode ', hscode_value)
   
    if diagram_type == 1:
        _query_str = 'select {} from Records where HSCode={}'.format(','.join(domain_subset), hscode_value)

    elif diagram_type == 2:

        _query_str = 'select ConsigneeName, HSCode, ShipperName from Records ' \
                     'JOIN ConsigneePanjivaID  on  ConsigneePanjivaID.ConsigneePanjivaID=Records.ConsigneePanjivaID ' \
                     'JOIN ShipperPanjivaID on ShipperPanjivaID.ShipperPanjivaID=Records.ShipperPanjivaID ' \
                     'where  HSCode={}'.format(hscode_value)
    print(_query_str)
    # ------------------------------
    # Use the sqllite  DB to read in all records that match HSCode
    # ------------------------------
    df = pd.read_sql(
        _query_str,
        con=SQL_Conn,
        index_col=None
    )

    return df

def get_sankey_diagram(
        PanjivaRecordID,
        DATA_LOC=None,
        subDIR=None,
        diagram_type=1
):
    # -----------------------
    # Read data
    # -----------------------

    df = obtain_data(
        PanjivaRecordID,
        DATA_LOC=DATA_LOC,
        subDIR=subDIR,
        diagram_type=1
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
    label2id = { e[1]:e[0] for e in enumerate(all_vals,0)}
    # replace in df
    df_1 = df.copy(deep=True)
    for col in list(df_1.columns):
        df_1[col] = df_1[col].parallel_apply( lambda x: label2id[x])

    source_list = []
    target_list = []
    value_list = []

    for a,b in zip(domain_subset[:-1],domain_subset[1:]):
       
        _df_ = df_1[[a,b]].groupby([a,b]).size().reset_index(name='count')
        source_list.extend(_df_[a].values.tolist())
        target_list.extend(_df_[b].values.tolist())
        value_list.extend(_df_['count'].values.tolist())
        print(len(target_list), len(source_list), len(value_list))
        
    if diagram_type == 1:
        title = 'Sankey Diagram'
    elif diagram_type == 2:
        title = 'Sankey Diagram'

    # -----------------
    # create a graph for sankey diagram
    # -----------------
  

    label_list = list(all_vals)
    label_list = [ _.split(':') for _ in label_list]
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
            source= link_source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
            target= link_target,
            value= value_list
        ))])

    fig.update_layout(
        title_text=title,
        font_size=15,
        paper_bgcolor='white'
    )
    return fig
