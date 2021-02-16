import os
import pandas as pd
import sys
import torch
sys.path.append('./..')
sys.path.append('./../..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
from ApE import model as ape_model
from MEAD import model_AD_1 as mead_model
import yaml
from common_utils import borda_count
import argparse


ID_COL = 'PanjivaRecordID'
DATA_LOC = './../generated_data_v1'
config = None
OUTPUT_DIR= './ensemble_output'

def get_domain_dims():
    global DATA_LOC
    global DIR
    with open(os.path.join(DATA_LOC, '{}/domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

def setup_up(DIR):
    global config
    global OUTPUT_DIR
    config_file = 'ensemble_config.yaml'
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    config = config[DIR]
    pathobj = Path(OUTPUT_DIR)
    pathobj.mkdir(exist_ok=True, parents=True)
    pathobj = Path(os.path.join(OUTPUT_DIR,DIR))
    pathobj.mkdir(exist_ok=True, parents=True)
    return


def main(
    include_negative_anomalies = False,
    print_details = False
):
    global DIR
    global OUTPUT_DIR
    global config
    global ID_COL
    global DATA_LOC
    global device
    test_df = pd.read_csv(
        os.path.join(DATA_LOC, '{}/stage_2/test_serialized.csv'.format(DIR)),
        index_col=None
    )
    domain_dims = get_domain_dims()
    entity_count = sum(domain_dims.values())
    id_list_normal = test_df[ID_COL].values.tolist()
    del test_df[ID_COL]
    test_x = test_df.values

    # -------------------------------------------------
    # Positive anomalies

    anomalies_src_path = os.path.join(DATA_LOC, 'generated_anomalies/{}'.format(DIR))
    test_df_p = pd.read_csv(os.path.join(anomalies_src_path, 'pos_anomalies.csv'), index_col=None)
    id_list_Pos = test_df_p[ID_COL].values.tolist()
    del test_df_p[ID_COL]
    test_x_Pos = test_df_p.values
    
    # -------------------------------------------------
    # Negative anomalies
    # -------------------------------------------------

    test_df_n = pd.read_csv(os.path.join(anomalies_src_path, 'neg_anomalies.csv'), index_col=None)
    id_list_Neg = test_df_n[ID_COL].values.tolist()
    del test_df_n[ID_COL]

    test_x_Neg = test_df_n.values
    # scores_3 = model.score_samples(test_xn)
    
    label_list_normal = [0 for _ in range(test_x.shape[0])]
    label_list_p = [1 for _ in range(test_x_Pos.shape[0])]
    label_list_n = [-1 for _ in range(test_x_Neg.shape[0])]
    checkpoints = [2.5,5,7.5]
    list_indiv_ranks = []
    for model_type, _list in config.items():
        model_container_obj =None
        if _list is None or len(_list)==0: 
            continue 
        if print_details :print(model_type)
        for _dict in _list:
            emb_dim = _dict['emb_dim']
            if model_type == 'ape':
                model_container_obj = ape_model.APE_container(
                    emb_dim=emb_dim,
                    device=device,
                    domain_dims=domain_dims
                )
                _path = os.path.join('ApE','saved_model', DIR, _dict['file'])
              
                model_container_obj.load_model(_path)

            elif model_type == 'mead':
                model_container_obj = mead_model.AD_model_container(
                    emb_dim=emb_dim,
                    device=device,
                    entity_count = entity_count
                )
                _path = os.path.join('MEAD','saved_model', DIR, _dict['file'])
               
                model_container_obj.load_model(_path)
            scores_normal = model_container_obj.predict(test_x)
            scores_pos_anom = model_container_obj.predict(test_x_Pos)
            scores_neg_anom = model_container_obj.predict(test_x_Neg)
            scores = scores_normal + scores_pos_anom + scores_neg_anom
            labels = label_list_normal + label_list_p + label_list_n
            id_list = id_list_normal + id_list_Pos + id_list_Neg
            
            if include_negative_anomalies is False:
                scores = scores_normal + scores_pos_anom  
                labels = label_list_normal + label_list_p 
                id_list = id_list_normal + id_list_Pos 
            
            data = {'label': labels, 'score': scores , 'PanjivaRecordID': id_list}
            df = pd.DataFrame(data)
            df = df.sort_values(by='score')

            # Check performance
            for k in checkpoints:
                tmp = df.head(int(df.shape[0]*k/100))
                prec1 = len(tmp.loc[tmp['label']!=0])/len(tmp)
                prec2 = len(tmp.loc[tmp['label']==1])/len(tmp)
                recall_1 = len(tmp.loc[tmp['label']!=0])/len(label_list_p)
                recall_2 = len(tmp.loc[tmp['label']==1])/len(label_list_p)
                if print_details :
                    print( k/100, 'Precision:', prec1, prec2 ,' | Recall :', recall_1,recall_2)
            
            if model_type == 'mead' or True:
                list_indiv_ranks.append(df[ID_COL].values.tolist())
    
    print('-----')
    combined_rank = borda_count.borda_count(list_indiv_ranks)
    combined_rank = combined_rank.rename(columns={'ID':ID_COL})
    labels = label_list_normal + label_list_p + label_list_n
    id_list = id_list_normal + id_list_Pos + id_list_Neg
    
    if include_negative_anomalies is False:
        labels = label_list_normal + label_list_p 
        id_list = id_list_normal + id_list_Pos 
        
            
    data = { 'label': labels,  'PanjivaRecordID': id_list}
    _df = pd.DataFrame(data)
    _df = _df.merge(combined_rank, on=[ID_COL], how ='inner')
    _df = _df.sort_values(by='rank', ascending=False)
    for k in checkpoints:
        tmp = _df.head(int(_df.shape[0]*k/100))
        prec1 = len(tmp.loc[tmp['label']!=0])/len(tmp)
        prec2 = len(tmp.loc[tmp['label']==1])/len(tmp)
        recall_1 = len(tmp.loc[tmp['label']!=0])/len(label_list_p)
        recall_2 = len(tmp.loc[tmp['label']==1])/len(label_list_p)
        print( k/100, 'Precision:', prec1, prec2 ,' | Recall :', recall_1,recall_2)
    
    _df.to_csv(os.path.join(OUTPUT_DIR,DIR,'AD_output.csv'),index=None)  
        

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5'],
    default='us_import1'
)

parser.add_argument(
    '--include_neg_anom',
    action='store_true'
)
parser.add_argument(
    '--print_details',
    action='store_true'
)
args = parser.parse_args()
print_details = args.print_details
include_neg_anom = args.include_neg_anom
DIR = args.DIR
setup_up(DIR)

main(include_neg_anom, print_details)