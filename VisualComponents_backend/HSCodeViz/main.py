import pandas as pd

import plotly.express as px
import pandas as pd
import sys
import os
import plotly
import plotly.graph_objects as go
import plotly.io as pio
sys.path.append('./../')
sys.path.append('./../..')
import pickle
from DB_Ingestion.sqlite_engine import sqlite
SQL_conn =None
DATA_LOC = None
subDIR = None


def initialize(
    _DATA_LOC,
    _subDIR
):
    global DATA_LOC, subDIR
    global SQL_conn
    SQL_conn = sqlite().get_engine()
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    
    return 


# ========================================
# Return Radial sunburst chart for the HSCode breakdown of Consignee and Shipper Respectively
# In viz set header for Panel as 'HS Code distribution for Consignee' 'HS Code distribution for Shipper'
# ========================================
def get_HSCode_distribution(
    record_id, 
    return_type=2
):
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
    

    if return_type == 1: 
        return fig_c,fig_s
    elif return_type == 2:
        fig_html_c = pio.to_html(fig_c, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        fig_html_s = pio.to_html(fig_s, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        return fig_html_c, fig_html_s
    
# ---------------------------------

# initialize(
#     _DATA_LOC='./../generated_data_v1/us_import',
#     _subDIR='01_2016'
# )

# record_id = '121983256'
# record_id = '121092583'
