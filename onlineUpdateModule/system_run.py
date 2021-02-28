#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from sklearn.preprocessing import normalize
sys.path.append('./.')
sys.path.append('./..')
from pathlib import Path
import argparse
import pickle
import copy
import json
from onlineGD import onlineGD
from loss_function_grad import maxDotProd_gradient, calculate_cosineDist_gradient
from linearModel import linearClassifier_bEF
from record import record_class
import yaml
import time
from collections import OrderedDict
from common_utils import utils
from sklearn.utils import shuffle

explantions_file_path = None
embedding_data_path = None
serialID_mapping_loc = None
anomalies_pos_fpath = None
anomalies_neg_fpath = None
feedback_batch_size = None
top_K_count = None
interaction_type = 'concat'

'''
embedding_data_path  = './../../createGraph_trade/saved_model_data/{}'.format(DIR)
serialID_mapping_loc = './../../generated_data_v1/{}/idMapping.csv'.format(DIR)
anomalies_pos_fpath = './../../generated_data_v1/generated_anomalies/{}/pos_anomalies.csv'.format(DIR)
anomalies_neg_fpath = './../../generated_data_v1/generated_anomalies/{}/neg_anomalies.csv'.format(DIR)
explantions_f_path =  './../../generated_data_v1/generated_anomalies/{}/pos_anomalies_explanations.json'.format(DIR)
'''


def setup_config(DIR):
    global explantions_file_path
    global embedding_data_path
    global serialID_mapping_loc
    global anomalies_pos_fpath
    global anomalies_neg_fpath
    global domain_dims
    global test_data_serialized_loc
    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    serialID_mapping_loc = config['serialID_mapping_loc'].format(DIR)
    embedding_data_path = config['embedding_data_path'].format(DIR)
    explantions_file_path = config['explantions_file_path'].format(DIR)
    anomalies_pos_fpath = config['anomalies_pos_fpath'].format(DIR)
    anomalies_neg_fpath = config['anomalies_neg_fpath'].format(DIR)
    test_data_serialized_loc = config['test_data_serialized_loc'].format(DIR)

    with open(config['domain_dims_file_path'].format(DIR), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return


# ---------------------------------------------------------------------------------
def get_serialID_to_entityID():
    global serialID_mapping_loc
    idMapper_file = os.path.join(serialID_mapping_loc)
    mapping_df = pd.read_csv(idMapper_file, index_col=None)
    serialID_to_entityID = {}

    for i, row in mapping_df.iterrows():
        serialID_to_entityID[row['serial_id']] = row['entity_id']
    return serialID_to_entityID


# ---------------------------
# Get records which are deemed nominal/normal
# ---------------------------
def obtain_normal_samples():
    global test_data_serialized_loc
    normal_data = pd.read_csv(
        test_data_serialized_loc, index_col=None
    )

    _df = normal_data.sample(5000)
    obj_list = []
    for i in tqdm(range(_df.shape[0])):
        obj = record_class(_df.iloc[i].to_dict(), -1)
        obj_list.append(obj)
    data_x = []
    for _obj in obj_list:
        data_x.append(_obj.x)
    data_x = np.stack(data_x)
    return data_x


def get_trained_classifier(X, y, num_domains, emb_dim, num_epochs=10000):
    global domain_dims
    global interaction_type
    classifier_obj = linearClassifier_bEF(
        num_domains=num_domains,
        emb_dim=emb_dim,
        num_epochs=num_epochs,
        L2_reg_lambda=0.0025,
        force_reg=False,
        interaction_type=interaction_type
    )

    classifier_obj.setup_binaryFeatures(
        domain_dims,
        binaryF_domains=['ConsigneePanjivaID', 'ShipperPanjivaID']
    )

    # classifier_obj.fit_on_pos(X, np.ones(X.shape[0]),n_epochs=10000)
    classifier_obj.fit(X, y, log_interval=5000)
    classifier_obj.fit_on_pos(X, y, n_epochs=num_epochs // 2, log_interval=1000)
    return classifier_obj


def fetch_entityID_arr_byList(data_df, id_list):
    global domain_dims
    domain_list = list(domain_dims.keys())
    ID_COL = 'PanjivaRecordID'
    data_df = data_df.copy(deep=True)
    data_df = data_df.loc[data_df[ID_COL].isin(id_list)]
    # Order of id_list has to be preserved!!!
    X = []
    for _id in id_list:
        _tmp = data_df.loc[data_df[ID_COL] == _id][domain_list].iloc[0].values.tolist()
        X.append(_tmp)
    return np.array(X).astype(int)


def execute_with_input(
        clf_obj,
        working_df,
        ref_data_df,
        domainInteraction_index,
        num_coeff,
        emb_dim,
        data_ID_to_matrix,
        check_next_values=[10, 20, 30, 40, 50],
        batch_size=10
):
    global domain_dims
    global interaction_type
    ID_COL = 'PanjivaRecordID'
    BATCH_SIZE = batch_size
    working_df['delta'] = 0
    OGD_obj = onlineGD(num_coeff, emb_dim, calculate_cosineDist_gradient, interaction_type=interaction_type)
    W = clf_obj.W.cpu().data.numpy()
    OGD_obj.set_original_W(W)

    max_num_batches = len(working_df) // BATCH_SIZE + 1
    acc = []
    recall = []
    domain_list = list(domain_dims.keys())
    discovered_count = [0.0]
    total_relCount = len(working_df.loc[working_df['label'] == 1])

    # -------------------------------------------------
    #  Main loop
    # -------------------------------------------------
    next_K_precision = []

    for batch_idx in tqdm(range(max_num_batches)):
        cur = working_df.head(BATCH_SIZE)
        flags = []  # Whether a pos anomaly or not
        terms = []  # Explanation terms

        # Count( of discovered in the current batch ( at the top; defined by batch size )
        cum_cur_discovered = discovered_count[-1] + len(cur.loc[cur['label'] == 1])
        _recall = float(cum_cur_discovered) / total_relCount
        discovered_count.append(
            cum_cur_discovered
        )
        recall.append(_recall)
        x_ij = []
        x_entityIds = []

        for i, row in cur.iterrows():
            _mask = np.zeros(len(domainInteraction_index))
            if row['label'] == 1:
                _mask[row['expl_1']] = 1
                _mask[row['expl_2']] = 1
                flags.append(1)
                terms.append((row['expl_1'], row['expl_2'],))
            else:
                flags.append(0)
                terms.append(())
            id_value = row['PanjivaRecordID']
            x_ij.append(data_ID_to_matrix[id_value])

            row_dict = ref_data_df.loc[(ref_data_df[ID_COL] == id_value)].iloc[0].to_dict()
            x_entityIds.append([row_dict[d] for d in domain_list])

        x_entityIds = np.array(x_entityIds)
        x_ij = np.array(x_ij)

        final_gradient, _W = OGD_obj.update_weight(
            flags,
            terms,
            x_ij
        )

        # ----------------------------------------------------
        # Update Model
        # ----------------------------------------------------
        clf_obj.update_W(_W)
        clf_obj.update_binary_VarW(x_entityIds, flags)

        _tail_count = len(working_df) - BATCH_SIZE
        working_df = working_df.tail(_tail_count).reset_index(drop=True)

        if len(working_df) == 0:
            break

        # Obtain scores
        x_ij_test = []
        x_entityIds = fetch_entityID_arr_byList(
            ref_data_df,
            working_df['PanjivaRecordID'].values.tolist()
        )
        for _id in working_df['PanjivaRecordID'].values:
            x_ij_test.append(data_ID_to_matrix[_id])

        x_ij_test = np.array(x_ij_test)

        new_scores = clf_obj.predict_bEF(x_entityIds, x_ij_test)
        old_scores = working_df['cur_score'].values
        _delta = new_scores - old_scores
        working_df['delta'] = _delta
        working_df = working_df.sort_values(by='delta', ascending=False)
        working_df = working_df.reset_index(drop=True)
        _next_K_precision = []
        for check_next in check_next_values:
            tmp = working_df.head(check_next)
            _labels = tmp['label'].values
            res = len(np.where(_labels == 1)[0])
            _precison = res / check_next
            _next_K_precision.append(_precison)
        next_K_precision.append(_next_K_precision)
    return next_K_precision, recall


def execute_without_input(
        working_df,
        check_next_values=[10, 20, 30, 40, 50],
        batch_size=10
):
    BATCH_SIZE = batch_size
    working_df['delta'] = 0

    num_batches = len(working_df) // BATCH_SIZE + 1
    next_K_precision = []
    recall = []
    discovered_count = [0.0]
    total_relCount = len(working_df.loc[working_df['label'] == 1])

    for b in range(num_batches):
        cur = working_df.head(BATCH_SIZE)
        cum_cur_discovered = discovered_count[-1] + len(cur.loc[cur['label'] == 1])
        _recall = float(cum_cur_discovered) / total_relCount
        discovered_count.append(
            cum_cur_discovered
        )
        recall.append(_recall)

        working_df = working_df.iloc[BATCH_SIZE:]
        working_df = working_df.reset_index(drop=True)
        _next_K_precision = []
        for check_next in check_next_values:
            tmp = working_df.head(check_next)
            _labels = tmp['label'].values
            res = len(np.where(_labels == 1)[0])
            precision = res / check_next
            _next_K_precision.append(precision)
        next_K_precision.append(_next_K_precision)
    return next_K_precision, recall


def main_executor():
    global explantions_file_path
    global embedding_data_path
    global serialID_mapping_loc
    global anomalies_pos_fpath
    global anomalies_neg_fpath
    global domain_dims
    global test_data_serialized_loc
    global feedback_batch_size
    global DIR

    # ============================================

    anom_pos_df = pd.read_csv(anomalies_pos_fpath, index_col=None)
    anom_neg_df = pd.read_csv(anomalies_neg_fpath, index_col=None)

    # ============================================
    # setup objects

    serialID_to_entityID = get_serialID_to_entityID()
    record_class.__setup_embedding__(embedding_data_path, serialID_to_entityID, _normalize=True)
    emb_dim = record_class.embedding['HSCode'].shape[1]

    # main_data_df has the records with entity ids
    main_data_df = pd.concat([anom_pos_df, anom_neg_df], axis=0)
    main_data_df = utils.convert_to_UnSerializedID_format(main_data_df, DIR)
    # -------------------------------------------
    obj_list = []
    for i in tqdm(range(anom_neg_df.shape[0])):
        obj = record_class(anom_neg_df.iloc[i].to_dict(), -1)
        obj_list.append(obj)

    for i in tqdm(range(anom_pos_df.shape[0])):
        obj = record_class(anom_pos_df.iloc[i].to_dict(), 1)
        obj_list.append(obj)

    # Read in the explantions
    with open(explantions_file_path, 'rb') as fh:
        explanations = json.load(fh)

    explanations = {int(k): [sorted(_) for _ in v] for k, v in explanations.items()}
    num_domains = len(domain_dims)
    domain_idx = {e[0]: e[1] for e in enumerate(domain_dims.keys())}

    domainInteraction_index = {}
    k = 0
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            domainInteraction_index['_'.join((domain_idx[i], domain_idx[j]))] = k
            k += 1

    data_x = []
    data_id = []
    data_label = []
    data_ID_to_matrix = {}

    for _obj in obj_list:
        data_x.append(_obj.x)
        data_id.append(_obj.id)
        data_label.append(_obj.label)
        data_ID_to_matrix[_obj.id] = _obj.x

    data_x = np.stack(data_x)
    data_label = np.array(data_label)
    data_id = np.array(data_id)

    idx = np.arange(len(data_id), dtype=int)
    np.random.shuffle(idx)

    data_x = data_x[idx]
    data_label = data_label[idx]
    data_id = data_id[idx]

    X_0 = data_x  # Relevant anomalies
    X_1 = obtain_normal_samples()  # Nominal
    y_0 = np.ones(X_0.shape[0])
    y_1 = -1 * np.ones(X_1.shape[0])
    y = np.hstack([y_0, y_1])
    X = np.vstack([X_0, X_1])
    num_coeff = len(domainInteraction_index)
    classifier_obj = get_trained_classifier(X, y, num_domains, emb_dim)
    W = classifier_obj.W.cpu().data.numpy()
    emb_dim = W.shape[-1]

    # classifier_obj.predict_score_op(X_0)
    # Create a reference dataframe  :: data_reference_df
    data_reference_df = pd.DataFrame(
        data=np.vstack([data_id, data_label]).transpose(),
        columns=['PanjivaRecordID', 'label']
    )

    data_reference_df['baseID'] = data_reference_df['PanjivaRecordID'].apply(lambda x: str(x)[:-3])
    data_reference_df['expl_1'] = -1
    data_reference_df['expl_2'] = -1
    data_reference_df['original_score'] = 1

    for i, row in data_reference_df.iterrows():
        _id = int(row['PanjivaRecordID'])
        if _id in explanations.keys():
            entry = explanations[_id]
            domain_1 = entry[0][0]
            domain_2 = entry[0][1]
            data_reference_df.loc[i, 'expl_1'] = domainInteraction_index['_'.join(sorted([domain_1, domain_2]))]
            domain_1 = entry[1][0]
            domain_2 = entry[1][1]
            data_reference_df.loc[i, 'expl_2'] = domainInteraction_index['_'.join(sorted([domain_1, domain_2]))]
        _x = data_ID_to_matrix[_id]
        data_reference_df.loc[i, 'original_score'] = classifier_obj.predict_score_op(np.array([_x]))[0]

    data_reference_df['cur_score'] = data_reference_df['original_score'].values

    # To get random results
    # Randomization
    cur_df = data_reference_df.copy()
    cur_df = cur_df.sample(frac=1).reset_index(drop=True)
    cur_df = shuffle(cur_df).reset_index(drop=True)
    check_next_values = [10, 20, 30, 40, 50]
    next_K_precision_wI, recall_wI = execute_with_input(
        clf_obj=copy.deepcopy(classifier_obj),
        working_df=cur_df,
        ref_data_df=main_data_df,
        domainInteraction_index=domainInteraction_index,
        num_coeff=num_coeff,
        emb_dim=emb_dim,
        data_ID_to_matrix=data_ID_to_matrix,
        check_next_values=check_next_values,
        batch_size=feedback_batch_size
    )

    next_K_precision_nI, recall_nI = execute_without_input(
        working_df=cur_df,
        batch_size=feedback_batch_size
    )

    def aux_create_result_df(
            next_K_precision, recall, check_next_values
    ):

        next_K_precision = np.array(next_K_precision)
        recall = np.array(recall)
        idx = np.arange(max(recall.shape[0], next_K_precision.shape[0]))

        # fill in zeros
        if next_K_precision.shape[0] < idx.shape[0]:
            pad = np.zeros([idx.shape[0] - next_K_precision.shape[0], next_K_precision.shape[1]])
            next_K_precision = np.concatenate([next_K_precision, pad], axis=0)

        columns = ['idx'] + ['recall'] + ['Prec@next_' + str(_) for _ in check_next_values]
        _data = np.concatenate([
            idx.reshape([-1, 1]),
            recall.reshape([-1, 1]),
            next_K_precision,
        ], axis=-1)
        result_df = pd.DataFrame(
            _data, columns=columns
        )
        return result_df

    results_with_input = aux_create_result_df(next_K_precision_wI, recall_wI, check_next_values)
    results_no_input = aux_create_result_df(next_K_precision_nI, recall_nI, check_next_values)
    return results_with_input, results_no_input

