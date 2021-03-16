import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
import numpy as np
import pandas as pd
sys.path.append('./../..')
from redisStore import redisUtil
from pathlib import Path
import plotly
from plotly import express as px
import os
import plotly.io as pio
import pickle
from DB_Ingestion.sqlite_engine import sqlite
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sqlalchemy import create_engine
from plotly import io
from plotly import express as px
from plotly.graph_objs import Layout

DATA_LOC = None
subDIR = None
# Singleton object
redis_obj = redisUtil.redisStore
EMB_DIM = None
htmlSaveDir = None
SQL_Conn = None
reference_df = None

# =============================
#  This needs to be called to ingest data into redis,
# =============================
def initialize(
    _DATA_LOC,
    _subDIR,
    mp2v_emb_dir = './../records2graph/saved_model_data',
    emb_dim = 64,
    _htmlSaveDir = None
):
    global redis_obj
    global DATA_LOC
    global subDIR
    global EMB_DIM
    global htmlSaveDir
    global SQL_Conn
    global reference_df
    
    EMB_DIM = emb_dim
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    
    df_1 = pd.read_csv(os.path.join(DATA_LOC,subDIR,'train_data.csv'), index_col=None)
    df_2 = pd.read_csv(os.path.join(DATA_LOC,subDIR,'test_data.csv'), index_col=None)
    reference_df = df_1.append(df_2,ignore_index=True)
    
    SQL_Conn = sqlite.get_engine()
    if _htmlSaveDir is None:
        htmlSaveDir = './htmlCache'
    else:
        htmlSaveDir = _htmlSaveDir
    Path(htmlSaveDir).mkdir(exist_ok=True,parents=True )
    
    redis_obj.ingest_record_data(
        DATA_LOC=DATA_LOC,
        subDIR=subDIR
    )
    
    redis_obj.ingest_MP2V_embeddings(
        DATA_LOC,
        subDIR ,
        mp2v_emb_dir,
        emb_dim=emb_dim
    )
    return 


# --------------------
# Helper function
# --------------------
def get_comparison_vecotors(record_row, domain, entity_list):
    global redis_obj
    global EMB_DIM
    
    emb_dim = EMB_DIM
    vec = []
    for entity_id in entity_list:
        key = 'mp2v_{}_{}_{}'.format(emb_dim, domain, entity_id)
        vec.append(redis_obj.fetch_np(key))
    
    key = 'mp2v_{}_{}_{}'.format(emb_dim, domain,record_row[domain])
    target_entity_vec = redis_obj.fetch_np(key)
    vec.append(target_entity_vec)
    return (domain, np.array([target_entity_vec]), np.array(vec))


# ===========================================
# Return ::
# Dict of { <domain> : html_string of figure }
#
# ===========================================

def get_stackedComparisonPlots(
    record_id, 
    min_count = 1000,
    return_type=1
):
    global EMB_DIM
    global htmlSaveDir
    global SQL_Conn
    global DATA_LOC 
    global reference_df
    global subDIR 
    global redis_obj
    
    
    with open(os.path.join(DATA_LOC, subDIR,'col_val2id_dict.pkl'), 'rb') as fh:
            column_values2id = pickle.load(fh)
    column_id2value = { _domain: {v:k for k,v in _dict.items()} for _domain,_dict in column_values2id.items()  }

    with open(os.path.join(DATA_LOC, subDIR,'domain_dims.pkl'), 'rb') as fh:
            domain_dims = pickle.load(fh)

    record_row = redis_obj.fetch_data(record_id)
    print(record_row)
    record_row_w_vals = {} 
    for _domain in column_values2id.keys():
        record_row_w_vals[_domain] = column_id2value[_domain][record_row[_domain]]

    _query_str_0 = 'select PanjivaRecordID from Records where ConsigneePanjivaID={} and ShipperPanjivaID={}'.format(
        record_row_w_vals['ConsigneePanjivaID'],record_row_w_vals['ShipperPanjivaID']
    )
    _query_str_1 = 'select PanjivaRecordID from Records where ConsigneePanjivaID={} or ShipperPanjivaID={}'.format(
        record_row_w_vals['ConsigneePanjivaID'],record_row_w_vals['ShipperPanjivaID']
    )

    _query_str_6= 'select PanjivaRecordID from Records where PortOfLading="{}" and HSCode={} and  PortOfUnlading="{}"'.format(
        record_row_w_vals['PortOfLading'],record_row_w_vals['HSCode'], record_row_w_vals['PortOfUnlading']
    )
    _query_str_4 = 'select PanjivaRecordID from Records where ( ShipmentOrigin="{}" and PortOfLading="{}")  or (ShipmentDestination="{}" and PortOfUnlading="{}")'.format(
        record_row_w_vals['ShipmentOrigin'], 
        record_row_w_vals['PortOfUnlading'],
        record_row_w_vals['ShipmentDestination'],
        record_row_w_vals['PortOfLading'] 
    )

    _query_str_5 = 'select PanjivaRecordID from Records where ( ShipmentOrigin="{}" and HSCode={})  or (ShipmentDestination="{}" and HSCode={})'.format(
        record_row_w_vals['ShipmentOrigin'], 
        record_row_w_vals['HSCode'],
        record_row_w_vals['ShipmentDestination'],
        record_row_w_vals['HSCode'] 
    )

    _query_str_2 = 'select PanjivaRecordID from Records where ShipperPanjivaID={} and ShipmentOrigin="{}"'.format(
        record_row_w_vals['ShipperPanjivaID'], record_row_w_vals['ShipmentOrigin']
    )

    _query_str_3 = 'select PanjivaRecordID from Records where ConsigneePanjivaID={} and  ShipmentDestination="{}"'.format(
        record_row_w_vals['ConsigneePanjivaID'], record_row_w_vals['ShipmentDestination']
    )

    query_string_list = [_query_str_0, _query_str_1, _query_str_2,_query_str_3,  _query_str_4, _query_str_5, _query_str_6 ]


    data = None
    ID_COL = 'PanjivaRecordID'

    for _query in query_string_list:
        _df = pd.read_sql(
                _query,
                con=SQL_Conn,
                index_col=None
        )

        ids = _df[ID_COL].values.tolist()
        tmp = reference_df.loc[reference_df[ID_COL].isin(ids)]
        if data is None:
            data = tmp.copy()
        data = data.append(tmp,ignore_index=True)
        data = data.drop_duplicates(subset=[ID_COL])
        if len(data) >= min_count:
            data = data.head(min_count)
            break
        
    vectors_dict = {}
    for domain in domain_dims.keys():
        if domain in ['ConsigneePanjivaID','ShipperPanjivaID']:
            continue
        entity_list = data[domain].values.tolist()
        vectors = get_comparison_vecotors( record_row, domain, entity_list )
        vectors_dict[vectors[0]] = (vectors[1], vectors[2])

    # -----------------------------------------------------------------------------------
    fig = make_subplots(rows=2, cols=3, subplot_titles= list(vectors_dict.keys()))
    i = 1
    j = 1
    fig_dict = {}
    bg_colors = [
        'rgba(252, 240, 247,0.35)',
        'rgba(232, 244, 255,0.25)',
        'rgba(232, 255, 252,0.35)',
        'rgba(230, 248, 248,0.25)',
        'rgba(251, 255, 232,0.35)',
        'rgba(252, 251, 232,0.25)'
    ]
    i = 0
    for domain in domain_dims.keys():
        if domain not in vectors_dict.keys():
            continue
         
        target_entity_vec = vectors_dict[domain][0]
        _vectors = vectors_dict[domain][1]
        
        fig = px.density_contour(
            x = _vectors[:,0],
            y = _vectors[:,1]
        )

       
        fig.update_layout(showlegend=False)
        sub_figure = go.Scatter(
            x = _vectors[:,0],
            y = _vectors[:,1],
            mode = 'markers',
            marker = dict(
                color = 'rgb(163, 255, 71,0.20)',
                size = 10,
                line=dict(
                color='MediumPurple',
                    width=2
                )
            )
        )
        
        fig.add_trace(
            go.Scatter(
            x=target_entity_vec[:,0],
            y=target_entity_vec[:,1],
            mode="markers+text",
            marker = dict(
                color = 'rgba(230,0,10,0.95)',
                size = 15,
                line=dict(
                color='Yellow',
                    width=2
                )
            ),
            text=["Entity"],
            )
        )
        range_x = np.max(_vectors[:,0]) - np.min(_vectors[:,0])
        range_y = np.max(_vectors[:,1]) - np.min(_vectors[:,1])
        r1 = 0.04*range_x
        r2 = 0.04*range_y
        x0 =target_entity_vec[:,0][0]
        y0 =target_entity_vec[:,1][0]
        x1 = x0 - r1
        x2 = x0 + r1
        y1 = y0 - r2
        y2 = y0 + r2
        print(x1,x2,y1,y2)
        fig.add_shape(
            type="rect",
            x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict( width=2, color="rgba(230,0,10,0.95)"),
        )
            
        fig.add_trace(
            sub_figure
        )
        fig.update_xaxes(tickfont=dict(color='black', size=15))
        fig.update_yaxes(tickfont=dict(color='black', size=15))
        fig.update_layout(showlegend=False)
        
        
        layout = Layout(
            paper_bgcolor=bg_colors[i],
            plot_bgcolor='rgba(0,0,0,0.05)'
        )

        fig.update_layout(layout)
        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
        fig.update_layout(height=300, width=360)
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig_dict[domain] = fig
        i+=1
    if return_type==1:
        result = {}
        for domain,fig_obj in fig_dict.items():
            result[domain] = io.to_html(fig_obj, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        return result
    else:
        return fig_dict

     
# ===========================
# Example code
# ===========================

'''
initialize(
    _DATA_LOC = './../generated_data_v1/us_import',
    _subDIR = '01_2016',
    mp2v_emb_dir = './../records2graph/saved_model_data',
    emb_dim = 64,
    _htmlSaveDir = './htmlCache'
)


fig_dict = get_stackedComparisonPlots(
    record_id='121888536', 
    min_count = 1000, 
    return_type=1
)

'''



