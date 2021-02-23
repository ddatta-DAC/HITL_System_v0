import plotly
import plotly.express as px
import sys
import pandas as pd
import os
from plotly.graph_objs import *
import plotly.io as pio
from pathlib import Path
from redisStore import redisUtil
sys.path.append('./../..')
from DB_Ingestion.sqlite_engine import sqlite

SQL_conn = sqlite().get_engine()
DATA_LOC = None
subDIR = None
html_saveDir = None
json_saveDir = None

redis_obj = redisUtil.redisStore

def initialize(
        _DATA_LOC=None,
        _subDIR=None,
        _html_saveDir=None,
        _json_saveDir=None
):
    global redis_obj, DATA_LOC, subDIR, html_saveDir, json_saveDir
    
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    redis_obj.ingest_record_data(DATA_LOC, subDIR)

    if _html_saveDir is None:
        _html_saveDir = './htmlCache'
    html_saveDir = _html_saveDir
    Path(_html_saveDir).mkdir(exist_ok=True, parents=True)
    if _json_saveDir is None:
        json_saveDir = './jsonCache'
    else:
        json_saveDir = _json_saveDir

    Path(json_saveDir).mkdir(exist_ok=True, parents=True)
    return


'''
# ----------------------------------
Displays activity by month
If figure already created and use_cache ==True : fetch saved figure
Else create figure and store it in cache
return_type:
1 : Return plotly figure
2 : Return html
3 : Write to HTML and return the file name

# ----------------------------------
'''

def fetchTS(
        domain='ConsigneePanjivaID',
        _id=None,
        dateColumn='ArrivalDate',
        table='Records',
        title=None,
        use_cache=True,
        return_type=3
):
    global SQL_conn
    global DATA_LOC
    global subDIR
    global html_saveDir, json_saveDir
    
    signature = 'TimeSeries_{}_{}'.format(domain, id)
    json_fpath = os.path.join(json_saveDir, signature + '.json')
    fig = None

    if use_cache and os.path.exists(json_fpath):
        with open(json_fpath, 'r') as f:
            fig = pio.from_json(f.read())
    
    if fig is None:
        df = pd.read_sql(
            "select {},{} from {} where {}={}".format(domain, dateColumn, table, domain, _id), SQL_conn,
            index_col=None
        )

        df['ArrivalDate'] = pd.to_datetime(df['ArrivalDate'])
        df = df.reset_index(drop=True)
        df.index = df['ArrivalDate']
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['Date'] = df.apply(lambda x: str(x['year']) + '-' + str(x['month']), axis=1)
        del df['month']
        del df['year']

        fig = px.bar(
            df,
            x="Date",
            hover_data={"Date": "|%B "},
            title=title,
            color_continuous_scale= px.colors.sequential.Viridis,
        )
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b\n%Y",
            ticklabelmode="period"
        )
        fig.update_xaxes(rangeslider_visible=True)
        
        fig.update_layout(
            font_family="Courier New",
            font_color="DarkSlateGrey",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=16,
            paper_bgcolor='rgba(255, 255, 255 ,0.5)',
            plot_bgcolor='rgba(255,240,240,0.1)',
            xaxis_range=['2015-01-01','2017-12-31']
        )
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b %Y",
            ticklabelmode="period"
        )

        if use_cache:
            fig_json = pio.to_json(fig)
            fname = signature + '.json'
            with open(json_fpath, 'w') as f:
                f.write(fig_json)

    if return_type == 1:
        return fig
    elif return_type == 2:
        html_String = pio.to_html(fig, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        return html_String
    elif return_type == 3:
        fname = signature + '.html'
        fpath = os.path.join(html_saveDir, fname)
        fig.write_html(
            fpath, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False
        )
        return fpath
    return fig



def get_TimeSeries(
    PanjivaRecordID,
    return_type=3,
    use_cache=True
):
    global SQL_conn
    global DATA_LOC
    global subDIR
    
    df = pd.read_sql(
            "select {},{},{} from Records where {}={}".format('PanjivaRecordID', 'ConsigneePanjivaID', 'ShipperPanjivaID', 'PanjivaRecordID', str(int(PanjivaRecordID))), SQL_conn,
            index_col=None
    )
    consignee = str(int(df['ConsigneePanjivaID'].values[0]))
    shipper = str(int(df['ShipperPanjivaID'].values[0]))
    
    value_1 = fetchTS(
        domain='ConsigneePanjivaID',
        _id=consignee,
        title='Activity of Consignee',
        use_cache=use_cache,
        return_type=return_type
    )
    
    value_2 = fetchTS(
        domain='ShipperPanjivaID',
        _id=shipper,
        title='Activity of Shipper',
        use_cache=use_cache,
        return_type=return_type
    )
    return value_1, value_2


# ==================================
'''
initialize(
        _DATA_LOC='./../../',
        _subDIR=None,
        _html_saveDir=None,
        _json_saveDir=None
)
Sample calling code:
fetchTS(
        domain='ConsigneePanjivaID',
        id=None,
        dateColumn='',
        table='',
        title=None,
        use_cache=True,
        return_type=3
)

'''
