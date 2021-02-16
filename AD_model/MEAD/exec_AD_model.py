import os
import torch
import argparse
import pandas as pd
import numpy as np
try:
    from . import model_AD_1 as AD
except:
    import model_AD_1 as AD
import pickle
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import sys
sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')


# ===================================== #

def get_domain_dims(DIR):
    with open('./../../generated_data_v1/{}/domain_dims.pkl'.format(DIR), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

def get_training_data(DIR):
    x_pos = np.load('./../../generated_data_v1/{}/stage_2/train_x_pos.npy'.format(DIR))
    x_neg = np.load('./../../generated_data_v1/{}/stage_2/train_x_neg.npy'.format(DIR))
    return x_pos, x_neg

# ===================================== #

def main(DIR, saved_model):
    ID_COL = 'PanjivaRecordID'
    RESULTS_OP_PATH = 'AD_output'
    RESULTS_OP_PATH = os.path.join(RESULTS_OP_PATH,DIR)
    results_path = Path(RESULTS_OP_PATH)
    results_path.mkdir(parents=True,exist_ok=True)

    x_pos, x_neg = get_training_data(DIR)
    x_neg = x_neg.reshape([x_pos.shape[0], -1 , x_pos.shape[1]])
    domain_dims = get_domain_dims(DIR)
    total_entity_count = sum(domain_dims.values())
    model = AD.AD_model_container(total_entity_count, emb_dim=16, device=device)
    
    if saved_model is None:
        model.train_model(x_pos,x_neg, batch_size=128, epochs=50)
        model.save_model('saved_model/{}'.format(DIR))
    else:
        saved_model_path = os.path.join('./saved_model/{}/{}'.format(DIR, saved_model))
        model.load_model(saved_model_path)
    
    return

    model.model.mode='test'
    test_df = pd.read_csv(
        './../generated_data_v1/{}/stage_2/test_serialized.csv'.format(DIR),
        index_col=None
    )

    # -------------------------------------------------
    # Normal records
    id_list_normal = test_df[ID_COL].values.tolist()
    del test_df[ID_COL]
    test_x = test_df.values
    scores_1 = model.score_samples(test_x)

    # -------------------------------------------------
    # Positive anomalies

    anomalies_src_path = './../../generated_data_v1/generated_anomalies/{}'.format(DIR)
    test_df_p = pd.read_csv(os.path.join(anomalies_src_path, 'pos_anomalies.csv' ), index_col=None)
    id_list_p = test_df_p[ID_COL].values.tolist()
    del test_df_p[ID_COL]
    test_xp = test_df_p.values
    scores_2 =  model.score_samples(test_xp)

    # -------------------------------------------------
    # Negative anomalies
    
    test_df_n = pd.read_csv(os.path.join(anomalies_src_path, 'neg_anomalies.csv' ), index_col=None)
    id_list_n = test_df_n[ID_COL].values.tolist()
    del test_df_n[ID_COL]
    test_xn = test_df_n.values
    scores_3 = model.score_samples(test_xn)

    try:
        box = plt.boxplot([scores_1,scores_2,scores_3], notch=True, patch_artist=True)
        colors = ['cyan', 'pink', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.show()
    except:
        pass

    label_list_normal = [0 for _ in range(len(scores_1))]
    label_list_p = [1 for _ in range(len(scores_2))]
    label_list_n = [-1 for _ in range(len(scores_3))]
    scores = scores_1 + scores_2 + scores_3
    id_list = id_list_normal + id_list_p + id_list_n
    labels = label_list_normal + label_list_p + label_list_n
    data = {'label': labels, 'score': scores , 'PanjivaRecordID': id_list}
    df = pd.DataFrame(data)
    # Save data

    df = df.sort_values(by='score')
    df.to_csv(os.path.join(RESULTS_OP_PATH, 'AD_output.csv'), index=None)
    # ======
    # Print out some checks
    # ======
    for threshold in  [10, 100,250,500, 1000]:
        tmp = df.head(threshold)
        print( " Count of Anomalies {} | Pos {} | Negative {} | in top {} ".format(
            len(tmp.loc[tmp['label'] != 0]),
            len(tmp.loc[tmp['label'] == 1]),
            len(tmp.loc[tmp['label'] == -1]), threshold)
        )
    return

# ===================================== #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5'],
    default='us_import1'
)

parser.add_argument(
    '--saved_model',
    default=None
)

args = parser.parse_args()
DIR = args.DIR
saved_model = args.saved_model
main(DIR,saved_model)