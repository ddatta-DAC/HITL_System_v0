import pandas as pd
import numpy as np
import sys
sys.path.append('./../..')
from redisStore import redisUtil
from pathlib import Path
import plotly
from plotly import express as px
import multiprocessing as MP
from joblib import Parallel,delayed
import os
import plotly.io as pio
from tqdm import tqdm

DATA_LOC = None
subDIR = None
# Singleton object
redis_obj = redisUtil.redisStore
EMB_DIM = None
htmlSaveDir = None
anomaly_result_dir = None
ID_col = 'PanjivaRecordID'

# =============================
#  This needs to be called to ingest data into redis,
# =============================
def initialize(
    _DATA_LOC,
    _subDIR,
    mp2v_emb_dir = './../records2graph/saved_model_data',
    emb_dim = 64,
    _htmlSaveDir = './htmlCache',
    _anomaly_result_dir = None
):
    global redis_obj
    global DATA_LOC
    global subDIR
    global EMB_DIM
    global htmlSaveDir
    global anomaly_result_dir
    
    EMB_DIM = emb_dim
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    if _htmlSaveDir is None:
        htmlSaveDir = './htmlCache'
    else:
        htmlSaveDir = _htmlSaveDir
    htmlSaveDir = htmlSaveDir + '_' + _subDIR
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
    anomaly_result_dir = _anomaly_result_dir
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
    samples = samples.sort_values(by='score', ascending=False)
    count = min(len(samples) // 3, count)
    df = samples.head(count).copy(deep=True)
    del df['score']
    recordID_list = df[ID_col].astype(int).tolist()
    
    for id in tqdm(recordID_list):
        get_record_entityEmbeddings(int(id))
#     cpu_count = min(10, cpu_count)
#     Parallel(n_jobs=cpu_count, prefer="threads")(
#         delayed(get_record_entityEmbeddings)(int(id)) for id in tqdm(recordID_list)
#     )
    return  
    

'''
Main function 
'''
def get_record_entityEmbeddings(
    PanjivaRecordID,
    return_type = 2
):
    global htmlSaveDir
    
    signature = 'EmbeddingViz_all_{}'.format(str(int(PanjivaRecordID)))
    fname = signature + '.html'
    fpath = os.path.join(htmlSaveDir, fname)
    if os.path.exists(fpath):
        if return_type == 2:
            with open(fpath,'r') as fh:
                html_String = fh.read()
                return html_String
        elif return_type == 3:
            return fpath
    
    record = redis_obj.fetch_data(
        str(int(PanjivaRecordID))
    )
    data_dict = {}
    for _domain, entity_id in record.items():
        key = 'mp2v_{}_{}_{}'.format(EMB_DIM, _domain, entity_id)
        vec = redis_obj.fetch_np(key)
        if vec is not None:
            data_dict[_domain] = vec
            
    title_text = 'Emmbedding visualization of the Entities of the Record ID {} in 2-Dimension'.format(int(PanjivaRecordID))
    
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index().rename(columns={'index':'Domain', 0 :'X-component', 1 :'Y-component'})
    df['Domain'] = df['Domain'] .apply(lambda x: x.replace('PanjivaID',''))
    df['size'] = 12
    
    fig = px.scatter(df, x= 'X-component', y= 'Y-component', color="Domain", text="Domain", size='size')
    fig.update_traces(
        textposition='top center',
        mode="markers+text", 
        marker_line_width=2, 
        marker_size=15, 
        textfont_size=18
    )

    fig.update_layout(
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=0.98,
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=14.5,
                color="black"
            ),
            bgcolor='rgba(50,205,50,0.15)',
            bordercolor="Black",
            borderwidth=2
        ),
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="Black"
        ),
        plot_bgcolor='rgb(250,250,250)'
    )

    if return_type == 1:
        return df
    elif return_type == 2:
        html_String = pio.to_html(fig, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        fname = signature + '.html'
        fpath = os.path.join(htmlSaveDir, fname)
        fig.write_html(fpath, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        return html_String
    elif return_type == 3:
        fname = signature + '.html'
        fpath = os.path.join(htmlSaveDir, fname)
        fig.write_html(fpath, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
        return fpath
    return 

'''
Example:
main.initialize(
    _DATA_LOC='./../generated_data_v1/us_import',
    _subDIR='01_2016',
    mp2v_emb_dir = './../records2graph/saved_model_data',
    emb_dim = 64,
    _htmlSaveDir = './EmbViz_all/htmlCache',
    _anomaly_result_dir = './../AD_model/combined_output'
)
figpath = emb.get_record_entityEmbeddings(
    PanjivaRecordID= '121983692',
    return_type=3
)
'''










