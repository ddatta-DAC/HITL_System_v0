import pandas as pd
import numpy as np
import sys
sys.path.append('./../..')
from redisStore import redisUtil
from pathlib import Path
import plotly
from plotly import express as px
import os

DATA_LOC = None
subDIR = None
# Singleton object
redis_obj = redisUtil.redisStore
EMB_DIM = None
htmlSaveDir = None

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
    
    EMB_DIM = emb_dim
    DATA_LOC = _DATA_LOC
    subDIR = _subDIR
    if _htmlSaveDir is None:
        htmlSaveDir = './htmlCache'
    else:
        htmlSaveDir = _htmlSaveDir
    Path(htmlSaveDir).mkdir(exist_ok=True,parents=True )
    redisUtil.redisStore.ingest_record_data(
        DATA_LOC=DATA_LOC,
        subDIR=subDIR
    )
    
    redis_obj.ingest_MP2V_embeddings(
        DATA_LOC,
        subDIR ,
        mp2v_emb_dir,
        emb_dim=emb_dim
    )


def get_record_entityEmbeddings(
    PanjivaRecordID,
    return_type = 3
):
    global htmlSaveDir
    record = redis_obj.fetch_data(
        str(int(PanjivaRecordID))
    )
    signature = 'EmbeddingViz_all_{}'.format(str(int(PanjivaRecordID)))
    data_dict = {}
    for _domain, entity_id in record.items():
        key = 'mp2v_{}_{}_{}'.format(EMB_DIM, _domain, entity_id)
        vec = redis_obj.fetch_np(key)
        if vec is not None:
            data_dict[_domain] = vec
    
    title_text = 'Emmbedding visualization of the Entities of the Record ID {} in 2-Dimension'.format(int(PanjivaRecordID))
    
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index().rename(columns={'index':'domain', 0 :'X-component', 1 :'Y-component'})
    df['domain'] = df['domain'] .apply(lambda x: x.replace('PanjivaID',''))
    
    fig = px.scatter(df, x= 'X-component', y= 'Y-component', color="domain", text="domain" , title= title_text)
    fig.update_traces(textposition='top center', mode="markers+text", marker_line_width=2, marker_size=20, textfont_size=18)
                 
    if return_type == 1:
        return data_dict
    elif return_type == 2:
        html_String = pio.to_html(fig, include_plotlyjs='cdn', include_mathjax='cdn', full_html=False)
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
    _htmlSaveDir = './htmlCache'
)
figpath = emb.get_record_entityEmbeddings(
    PanjivaRecordID= '121983692',
    return_type=3
)
'''










