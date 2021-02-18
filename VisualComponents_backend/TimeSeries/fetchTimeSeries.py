import plotly
import plotly.express as px
import sys
import pandas as pd
import os
from plotly.graph_objs import *
import plotly.io as pio
sys.path.append('./../..')
from DB_Ingestion.sqlite_engine import sqlite

SQL_conn = sqlite().get_engine()
CACHE_dir = 'Cache'


# ----------------------------------
# Displays activity by month
# If figure already created and cache ==True : fetch saved figure
# Else create figure and store it in cache
# ----------------------------------

def fetchTS(domain='ConsigneePanjivaID', id=None, dateColumn='', table='', title=None, cache=True):
    global SQL_conn
    global CACHE_dir

    signature = '{}_{}'.format(domain, id)
    fname = os.path.join(CACHE_dir, signature + '.json')
    if cache and os.path.exists(fname):
        with open(fname, 'r') as f:
            fig = pio.from_json(f.read())
            return fig

    df = pd.read_sql("select {},{} from {} where {}={}".format(domain, dateColumn, table, domain, id), SQL_conn,
                     index_col=None)
    df['ArrivalDate'] = pd.to_datetime(df['ArrivalDate'])
    df = df.reset_index(drop=True)
    df.index = df['ArrivalDate']
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['Date'] = df.apply(lambda x: str(x['year']) + '-' + str(x['month']), axis=1)
    del df['month']
    del df['year']

    layout = Layout(
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)'
    )
    fig = px.bar(
        df,
        x="Date",
        hover_data={"Date": "|%B "},
        title=title
    )
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        ticklabelmode="period"
    )
    fig.update_xaxes(rangeslider_visible=True, )
    fig.update_layout(paper_bgcolor='rgb(255, 252, 260)')
    fig.update_layout(
        font_family="Courier New",
        font_color="DarkSlateGrey",
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green",
        font_size=15
    )

    if cache:
        fname = os.path.join(CACHE_dir, signature + '.json')
        #         fig.write_html(fname, include_plotlyjs='cdn')
        fig.write_json(fname)
    return fig
