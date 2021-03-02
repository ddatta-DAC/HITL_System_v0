#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import sys
import glob
from tqdm import tqdm

sys.path.append('./..')
sys.path.append('./../..')
from pathlib import Path
import argparse
import pickle
import copy
import json
from onlineUpdateModule.onlineGD import onlineGD
from onlineUpdateModule.loss_function_grad import calculate_cosineDist_gradient
from onlineUpdateModule.linearModel import linearClassifier_bEF
from onlineUpdateModule.record import record_class
import yaml
import time
from collections import OrderedDict
from utils import utils
from onlineUpdateModule.data_handler import data_handler

embedding_data_path = None
serialID_mapping_loc = None
anomalies_pos_fpath = None
anomalies_neg_fpath = None
feedback_batch_size = None
interaction_type = 'concat'

# '''
# embedding_data_path  = './../../createGraph_trade/saved_model_data/{}'.format(DIR)
# serialID_mapping_loc = './../../generated_data_v1/{}/idMapping.csv'.format(DIR)
# anomalies_pos_fpath = './../../generated_data_v1/generated_anomalies/{}/pos_anomalies.csv'.format(DIR)
# anomalies_neg_fpath = './../../generated_data_v1/generated_anomalies/{}/neg_anomalies.csv'.format(DIR)
# explantions_f_path =  './../../generated_data_v1/generated_anomalies/{}/pos_anomalies_explanations.json'.format(DIR)
# '''
# ------------------------


class onlineUpdateExecutor:
    def __init__(
            self,
            data_handler_object: data_handler,
            interaction_type='concat',
            update_at_every=10
    ):
        self.update_at_every = update_at_every
        self.feedback_input_counter = 0
        self.data_handler_object = data_handler_object
        self.interaction_type = interaction_type
        self.__build__()

    @staticmethod
    def __get_trained_classifier__(
            X,
            y,
            domain_dims,
            emb_dim,
            num_epochs=10000
    ):
        # TODO : Chanage n_epochs to 10000
        
        num_domains = len(domain_dims)
        classifier_obj = linearClassifier_bEF(
            num_domains=num_domains,
            emb_dim=emb_dim,
            num_epochs=num_epochs,
            L2_reg_lambda=0.001,
            force_reg=False,
            interaction_type=interaction_type
        )

        classifier_obj.setup_binaryFeatures(
            domain_dims,
            binaryF_domains=['ConsigneePanjivaID', 'ShipperPanjivaID']
        )

        # classifier_obj.fit_on_pos(X, np.ones(X.shape[0]),n_epochs=10000)
        
        classifier_obj.fit(X, y, log_interval=500)
        classifier_obj.fit_on_pos(X, y, n_epochs=num_epochs // 2, log_interval=500)
        return classifier_obj

    # ---------------------------
    # Get records which are deemed nominal/normal
    # ---------------------------
    @staticmethod
    def __obtain_normal_samples__(data_handler_obj: data_handler):
        _df = data_handler_obj.get_nominal_records_sample()
        obj_list = []
        for i in tqdm(range(_df.shape[0])):
            obj = record_class(_df.iloc[i].to_dict(), _label=-1, is_unserialized=True)
            obj_list.append(obj)
        data_x = []
        for _obj in obj_list:
            data_x.append(_obj.x)
        data_x = np.stack(data_x)
        return data_x
   
    def __build__(
            self
    ):
        data_handler_object = self.data_handler_object
        self.ID_COL = 'PanjivaRecordID'
        # -----------------------
        # Todo set up the inputs
        # -------------------------
        # data_handler_object =  data_handler()
        
        embedding_data_path = data_handler_object.embedding_data_path
        global serialID_mapping_loc

        # setup objects
        serialID_to_entityID = data_handler_object.get_serialID_to_entityID()
        record_class.__setup_embedding__(
            embedding_data_path,
            serialID_to_entityID,
            _normalize=True
        )
        emb_dim = record_class.embedding['HSCode'].shape[1]
        self.emb_dim = emb_dim
        print('Embedding Dimension', emb_dim)
        
        # main_data_df has the records with entity ids
        main_data_df = data_handler_object.get_working_records()
        
        # -------------------------------------------
        obj_list = []
        main_data_df[self.ID_COL] = main_data_df[self.ID_COL].astype(int)
        _tmp_ = main_data_df.copy(deep=True)
        del _tmp_['score']
        del _tmp_['label']
        for i in tqdm(range(_tmp_.shape[0])):
            obj = record_class(
                _tmp_.iloc[i].to_dict(), 
                _label=1, 
                is_unserialized=True
            )
            obj_list.append(obj)

        domain_dims = data_handler_object.domain_dims
        self.domain_dims = domain_dims
        num_domains = len(domain_dims)
        domain_idx = {e[0]: e[1] for e in enumerate(domain_dims.keys())}
        self.domain_idx = domain_idx
        self.domainInteraction_index = {}
        k = 0
        
        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                self.domainInteraction_index['_'.join((domain_idx[i], domain_idx[j]))] = k
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

        data_x = np.array(data_x)
        data_id = np.array(data_id)
        assert data_x.shape[0] == data_id.shape[0], print('Error !! check data set up pipeline')

        X_0 = data_x  # Relevant anomalies
        X_1 = onlineUpdateExecutor.__obtain_normal_samples__(data_handler_object)  # Nominal
        y_0 = np.ones(X_0.shape[0])
        y_1 = -1 * np.ones(X_1.shape[0])

        X = np.vstack([X_0, X_1])
        y = np.hstack([y_0, y_1])
        assert  X.shape[0] == y.shape[0], print('Error !! check data set up pipeline')

        self.num_coeff = len(self.domainInteraction_index)
        self.data_ID_to_matrix = data_ID_to_matrix
        # -------------------------------------------------------

        self.classifier_obj = onlineUpdateExecutor.__get_trained_classifier__(X, y, domain_dims, emb_dim)
        # Initial weights
        self.W_initial = self.classifier_obj.W.cpu().data.numpy()

        # Create a reference dataframe  :: data_reference_df
        data_reference_df = pd.DataFrame(
            data=data_id,
            columns=[self.ID_COL]
        )
        
        data_reference_df['original_score'] = 1
        for i, row in data_reference_df.iterrows():
            _id = int(row[self.ID_COL])
            _x = data_ID_to_matrix[_id]
            data_reference_df.loc[i, 'original_score'] = self.classifier_obj.predict_score_op(np.array([_x]))[0]

        data_reference_df['cur_score'] = data_reference_df['original_score'].values
        data_reference_df['delta'] = 0
        data_reference_df[self.ID_COL] = data_reference_df[self.ID_COL].astype(int)
         
        # This contains the record data 
        data_df = self.data_handler_object.data_df
        try:
            del data_df['label']
            del data_df['score']
        except:
            pass
        
        combined_df = data_reference_df.merge(
            data_df,
            on=self.ID_COL,
            how='inner',
        )
        combined_df[self.ID_COL] = combined_df[self.ID_COL].astype(int)
        self.data_reference_df = combined_df.copy(deep=True)
        self.working_data = combined_df.copy(deep=True)
        
        # ---------------------
        # label = 0 :: not labelled. Valid labels +1, -1
        # ---------------------
        self.working_data['label'] = 0
        self.data_reference_df['label'] = 0

        self.unprocessed_recordID_list = []
        self.OGD_obj = onlineGD(
            num_coeff=self.num_coeff,
            emb_dim=self.emb_dim,
            _gradient_fn=calculate_cosineDist_gradient,
            interaction_type=self.interaction_type,
            learning_rate= 0.05
        )
        self.OGD_obj.set_original_W(self.W_initial)
        self.labelled_recordID_list = []
        return

    # =======================
    # This is the main function to be called from outside to store input/feedback
    # Input format :
    # 1. PanjivaRecordID
    # 2. label provided ( +1/-1)
    # 3. pairs of entities that are the probable cause (if label is 1):
    # Format( [(domain_i,domain_j),....]
    # 4. entites flagged
    # Format( [domain_i, domain_j] ) :: Should be either consignee or shipper
    # =======================

    def register_feedback_input(self, recordID, label, entity_pair_list = [], entity_list = []):
        ID_COL = self.ID_COL
        
        for dom in entity_list:
            entity_id = self.data_handler_object.get_entity(self, recordID, dom)
            self.data_handler_object.store_entityFlag(
                domain=dom,
                entity_id=entity_id,
                label=label
            )
        
        self.data_handler_object.store_recordFlag(
            recordID,
            label
        )
        
        for entity_pair in entity_pair_list:
            entity_pair =  list(sorted(entity_pair))
            dom_1 = entity_pair[0]
            dom_2 = entity_pair[1]
            entity_id_1 = self.data_handler_object.get_entity( recordID, dom_1)
            entity_id_2 = self.data_handler_object.get_entity( recordID, dom_2)
            self.data_handler_object.register_entityInteraction(
                domain_1=dom_1,
                entity_1=entity_id_1,
                domain_2=dom_2,
                entity_2=entity_id_2
            )
            
        if recordID not in self.unprocessed_recordID_list:
            self.unprocessed_recordID_list.append(recordID)
            self.feedback_input_counter +=1
        
        if self.feedback_input_counter%self.update_at_every == 0:
            self.labelled_recordID_list.extend(self.unprocessed_recordID_list)
            print('[Info] Count of records labelled so far', len(self.labelled_recordID_list))
            # Update model
            print('[Info] Updating model start')
            self.__update_internal_model__()
            print('[Info] Updating model complete')
            # Clear the list to be used for model update

        self.working_data.loc[self.working_data[ID_COL]==recordID,'label'] = label
        self.data_reference_df.loc[self.data_reference_df[ID_COL] == recordID, 'label'] = label
        return

    # =======================
    # This is the main function to be called from outside to obtain current ranked list of anaomalies, which are not labelled
    # =======================
    def obtain_current_unlabelled_output(self):
        return self.working_data

    def obtain_current_labelled_output(self):
        return self.data_reference_df.loc[self.data_reference_df['label']!=0]
    
    def __reset_classifier__(self):
        self.classifier_obj.update_W(self.W_initial)
        return 
        
    # =================================================
    # This function is called when a the model has to be updated
    # =================================================
    def __update_internal_model__(
            self
    ):
        ID_COL = 'PanjivaRecordID'
        domain_list = list(self.domain_dims.keys())

        # -------------------------------------------------
        #  Obtain the set of records to use in  online GD
        # -------------------------------------------------
        cur_df = self.data_handler_object.get_working_records(
            exclude_updated = False
        )
        # Take out the records which have received feedback, but not been used to update model
        cur_df =  cur_df.loc[cur_df[ID_COL].isin(self.unprocessed_recordID_list)]

        flags = []  # Whether a relevant anomaly or not : Values = +1, -1(0)
        terms = []  # Explanation terms

        x_ij = []
        x_entityIds = []

        for i, row in cur_df.iterrows():
            recordID = row[ID_COL]
            _mask = np.zeros(len(self.domainInteraction_index))
            # Fetch the label
            if row['label'] == 1:
                domain_pairs = self.data_handler_object.get_domainPairInteractionsRegistered(recordID)
                _expl_terms = []
                for dp_key in domain_pairs:
                    _idx_ = self.domainInteraction_index[dp_key]
                    _mask[_idx_] = 1
                    _expl_terms.append(_idx_)
                terms.append(_expl_terms)
                flags.append(1)
            else:
                flags.append(0)
                terms.append(())
            id_value = int(row[self.ID_COL])
            x_ij.append(self.data_ID_to_matrix[id_value])

            row_dict = row.to_dict()
            x_entityIds.append([row[d] for d in domain_list])
        
        x_entityIds = np.array(x_entityIds).astype(int)
        x_ij = np.array(x_ij)        
        
        final_gradient, _W = self.OGD_obj.update_weight(
            flags,
            terms,
            x_ij
        )
        print('__update_internal_model__:: flags', flags)
        # ----------------------------------------------------
        # Update Model
        # ----------------------------------------------------
        self.classifier_obj.update_W(_W)
        self.classifier_obj.update_binary_VarW(
            x_entityIds,
            flags
        )

        # ------------------------------------------------------
        # Get predictions!
        # ------------------------------------------------------
        unmarked_recordIDs = self.working_data.loc[self.working_data['label']==0][ID_COL].values
        df = self.working_data.loc[self.working_data[ID_COL].isin(unmarked_recordIDs)]

        # Obtain scores
        x_ij_test = []
        x_entityIds = self.__fetch_entityID_arr_byList__(
            df,
            unmarked_recordIDs
        )
        for _id in unmarked_recordIDs:
            x_ij_test.append(self.data_ID_to_matrix[_id])

        x_ij_test = np.array(x_ij_test)

        new_scores = self.classifier_obj.predict_bEF(
            x_entityIds,
            x_ij_test
        )
        old_scores = df['cur_score'].values
        _delta = new_scores - old_scores
        df['delta'] = _delta

        # Place records at top which have had greatest change
        df = df.sort_values(by='delta', ascending=False)
        df = df.reset_index(drop=True)
        # -----------------------------
        # Update the working_data
        # -----------------------------
        self.working_data = df
        self.unprocessed_recordID_list = []
        

    def __fetch_entityID_arr_byList__(
            self,
            data_df,
            id_list
    ):

        domain_list = list(self.domain_dims.keys())
        ID_COL = 'PanjivaRecordID'
        data_df = data_df.copy(deep=True)
        data_df = data_df.loc[data_df[ID_COL].isin(id_list)]
        # Order of id_list has to be preserved!!!
        X = []
        for _id in id_list:
            _tmp = data_df.loc[data_df[ID_COL] == _id][domain_list].iloc[0].values.tolist()
            X.append(_tmp)
        return np.array(X).astype(int)


# ===========================

