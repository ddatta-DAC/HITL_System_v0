#!/usr/bin/env python
# coding: utf-8
import dgl
import torch
import sys
sys.path.append('./../..')
sys.path.append('./..')
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import yaml
import multiprocessing as MP
from utils import utils
from joblib import Parallel, delayed
import pickle
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.nn import functional as F
import dgl.function as fn
from torch import nn
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# Use training data for graph creation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================================
# GLOBALS
# ==============================================
# Margin for the triplet loss
MARGIN = 0.25

# ----------------------------
# First create a bipartite graph using training data
# 2 entity types : Consignee and Shipper
# Prefixes : 'Consignee' 'Shipper'
# ----------------------------
attr_consignee_prefix = 'ConsigneePanjivaID'
attr_shipper_prefix = 'ShipperPanjivaID'
bipartite_domains = sorted([attr_consignee_prefix, attr_shipper_prefix])
MP2V_features_LOC = None
SAVE_DIR = None
mp2vec_emb_dim = None
# ==============================================
# Functions
# ==============================================
def get_data_df(
        subDIR
):
    global attr_consignee_prefix
    global bipartite_domains
    global attr_shipper_prefix
    global DATA_LOC

    with open(os.path.join(DATA_LOC, subDIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    # This data does not have serial ID but domain specific ID
    data_df = pd.read_csv(
        os.path.join(DATA_LOC, subDIR,'train_data.csv'), index_col=None, low_memory=False
    )
    data_df = data_df.drop_duplicates(subset=list(domain_dims.keys()))
    # group by
    g_df = data_df.groupby([attr_consignee_prefix, attr_shipper_prefix]).size().reset_index(name='weight')

    # -----------------------------
    # Create synthetic mappiing
    # -----------------------------

    synID = 0
    cur = 0
    col_syn_id = []
    col_entity_id = []
    col_domain_names = []
    # ------------------
    for d in sorted(bipartite_domains):
        s = domain_dims[d]
        col_entity_id.extend(list(range(s)))
        col_domain_names.extend([d for _ in range(s)])
        tmp = np.arange(s) + cur
        tmp = tmp.tolist()
        col_syn_id.extend(tmp)
        cur += s

    data = {'domain': col_domain_names, 'entity_id': col_entity_id, 'syn_id': col_syn_id}
    synID_mapping_df = pd.DataFrame(data)

    # -------------------
    # Replace entity_id with synthetic id
    # -------------------
    mapping_dict = {}
    for domain in set(synID_mapping_df['domain']):
        tmp = synID_mapping_df.loc[(synID_mapping_df['domain'] == domain)]
        syn_id = tmp['syn_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        mapping_dict[domain] = {k: v for k, v in zip(entity_id, syn_id)}

    def convert_aux(val, domain):
        return mapping_dict[domain][val]

    for domain in tqdm(bipartite_domains):
        g_df[domain] = g_df[domain].parallel_apply(convert_aux, args=(domain,))
    return g_df, synID_mapping_df, data_df

# ------------------------------
'''
Core GNN 
'''
# ------------------------------
class GNN(torch.nn.Module):
    def __init__(
            self,
            inp_dim,
            out_dim,
            input_feature_label='mp2v',
            aggregator_type='mean',
            bias=True,
            norm=True,
            activation=None
    ):
        super(GNN, self).__init__()
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.aggregator_type = aggregator_type
        self.norm = norm
        self.activation = activation
        # ------
        # Assuming a 2 layer GNN
        # ------
        self.num_layers = 2
        self.input_feature_label = input_feature_label
        self.FC_w = nn.Linear(
            self.inp_dim * 3,
            self.out_dim,
            bias=bias
        )

        self.reset_parameters()
        self.aggregator_type = aggregator_type
        return

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(
            self.FC_w.weight,
            gain=gain
        )
        return

    def forward(self, graph_obj):

        for layer in range(self.num_layers):
            # --------------------------------------
            # If first layer initialize features with input features, then perform computation
            # --------------------------------------
            if layer == 0:
                # For the 1st layer initilize features with input feature
                graph_obj.ndata['features'] = graph_obj.ndata[self.input_feature_label]
                graph_obj.update_all(
                    fn.copy_u('features', 'm'),
                    fn.mean('m', 'h')
                )
            else:
                # ------------------------------
                # Update along the given edge
                # ------------------------------
                graph_obj.update_all(
                    fn.copy_u('features', 'm'),
                    fn.mean('m', 'h')
                )
                concat_ftrs = torch.cat([
                    graph_obj.ndata['features'],
                    graph_obj.ndata['h'],
                    graph_obj.ndata[self.input_feature_label]
                ],
                    dim=1
                )

                ftrs = self.FC_w(concat_ftrs)
                ftrs = F.tanh(ftrs)
                if self.norm:
                    ftrs = ftrs / torch.norm(ftrs, p=2)
                graph_obj.ndata['features'] = ftrs
        return


# --------------------------------------
# Triplet loss for training the GNN
# --------------------------------------
def triplet_loss(xt, xp, xn):
    global MARGIN
    m = MARGIN
    v1 = 1 - F.cosine_similarity(xt, xp, dim=1)
    v2 = 1 - F.cosine_similarity(xt, xn, dim=1)
    d = v1 - v2 + m
    loss = torch.max(d, torch.zeros_like(d))
    return torch.mean(loss, dim=0, keepdim=False)


def generate_pos_neg_neighbors(
        graph_obj
):
    # The source nodes
    nodes = graph_obj.nodes().cpu().data.numpy()
    edges = graph_obj.edges()
    node_src = edges[0].cpu().data.numpy()
    node_dest = edges[1].cpu().data.numpy()

    pos_nbrs = {_: [] for _ in nodes}
    neg_nbrs = {_: [] for _ in nodes}

    for i, j in zip(node_src, node_dest):
        pos_nbrs[i].append(j)

    # Stor the list of neighbors as values in the dictionary
    neg_nbr_dict = {}
    pos_nbr_dict = {}

    for node in nodes:
        pos_nbrs[node] = set(pos_nbrs[node])
        neg_nbrs[node] = list(set(nodes).difference(pos_nbrs[node]))
        neg_nbr_dict[node] = list(neg_nbrs[node])
        pos_nbr_dict[node] = list(pos_nbrs[node])

    return pos_nbr_dict, neg_nbr_dict


def generate_training_triplets(
        pos_nbr_dict,
        neg_nbr_dict
):
    nodes = list(pos_nbr_dict.keys())

    triplets = []

    def aux(n):
        return (n,
                np.random.choice(pos_nbr_dict[n], size=1)[0],
                np.random.choice(neg_nbr_dict[n], size=1)[0]
                )

    for n in nodes:
        t = aux(n)
        triplets.append(t)
    triplets = np.array(triplets)
    triplets = triplets[triplets[:, 0].argsort()]
    return triplets


def train_model(
        graph_obj,
        gnn_obj,
        pos_nbr_dict,
        neg_nbr_dict,
        num_epochs=10
):
    print(graph_obj)
    # Optimizer
    opt = torch.optim.Adam(
        list(gnn_obj.parameters())
    )

    pbar = tqdm(range(num_epochs))
    for e in pbar:
        opt.zero_grad()
        triplets = generate_training_triplets(pos_nbr_dict, neg_nbr_dict)

        # Forward
        gnn_obj(graph_obj)
        idx_t = LT(triplets[:, 0])
        emb_t = graph_obj.ndata['features'][idx_t, :]

        idx_p = LT(triplets[:, 1])
        idx_n = LT(triplets[:, 2])
        emb_p = graph_obj.ndata['features'][idx_p, :]
        emb_n = graph_obj.ndata['features'][idx_n, :]
        loss_val = triplet_loss(emb_t, emb_p, emb_n)
        loss_val.backward()
        opt.step()
        pbar.set_postfix({'Loss': '{:4f}'.format(np.mean(loss_val.cpu().data.numpy()))})
    return


# ------------------------------
# Main function
# ------------------------------

def exec_training(
    subDIR,
    train_epochs = 100
):
    global device
    global attr_consignee_prefix
    global bipartite_domains
    global attr_shipper_prefix
    global SAVE_DIR
    global MP2V_features_LOC
    global mp2vec_emb_dim
    SAVE_LOC = os.path.join(SAVE_DIR, subDIR)
    path_obj = Path(SAVE_LOC)
    path_obj.mkdir(exist_ok=True, parents=True)

    print('Device ', device)
    g_df, synID_mapping_df, data_df = get_data_df(subDIR)
    # ----------------------------------
    # Read In metapath2vec features
    # Those are stored as numpy arrays
    # ---------------------------------.
    mp2v_features = {}

    for attr in bipartite_domains:
        file = os.path.join(MP2V_features_LOC, subDIR, 'mp2v_{}_{}.npy'.format(attr,mp2vec_emb_dim))
        mp2v_features[attr] = np.load(file)

    '''
    To feed data into DGL graph 
    Create tensor of features
    '''
    input_emb_size = list(mp2v_features.values())[0].shape[1]
    num_entities = len(synID_mapping_df)
    input_features = np.zeros([num_entities, input_emb_size])

    for d in bipartite_domains:
        tmp = synID_mapping_df.loc[synID_mapping_df['domain'] == d]
        _syn_id = tmp['syn_id'].values.tolist()
        _entity_id = tmp['entity_id'].values.tolist()
        input_features[_syn_id] = mp2v_features[d][_entity_id]

    src = g_df[attr_consignee_prefix].values.tolist()
    dst = g_df[attr_shipper_prefix].values.tolist()
    nodes_src = src + dst
    nodes_dst = dst + src
    weights = g_df['weight'].values.tolist() + g_df['weight'].values.tolist()
    # DGL is not undirected by default
    graph_obj = dgl.graph((nodes_src, nodes_dst))
    graph_obj.edata['weight'] = FT(weights)
    graph_obj.ndata['mp2v'] = FT(input_features)
    graph_obj = graph_obj.to(device)

    print(' Graph object >> ', graph_obj)
    gnn_obj = GNN(
        inp_dim=input_features.shape[1],
        out_dim=32
    )

    gnn_obj = gnn_obj.to(device)
    pos_nbr_dict, neg_nbr_dict = generate_pos_neg_neighbors(graph_obj)

    train_model(
        graph_obj,
        gnn_obj,
        pos_nbr_dict,
        neg_nbr_dict,
        num_epochs=train_epochs
    )

    # Extract node features
    gnn_features = graph_obj.ndata['features'].cpu().data.numpy()
    # -------------------------------
    # Place feature in numpy arrays with entity_ID
    for d in bipartite_domains:
        tmp = synID_mapping_df.loc[synID_mapping_df['domain'] == d]
        _syn_id = tmp['syn_id'].values.tolist()
        _entity_id = tmp['entity_id'].values.tolist()
        x = np.zeros([len(tmp), gnn_features.shape[1]])
        x[_entity_id] = gnn_features[_syn_id]
        # Save the data
        file_name = '{}_gnn_{}.npy'.format(d, gnn_features.shape[1])
        file_path = os.path.join(SAVE_LOC, subDIR, file_name)
        np.save(file_path, x)
    return gnn_features


with open('config.yaml', 'r') as fh:
    config = yaml.safe_load(fh)
DATA_LOC = config['DATA_LOC']
train_epochs = int(config['train_epochs'])
SAVE_DIR = config['SAVE_DIR']
MP2V_features_LOC = config['MP2V_features_LOC']
mp2vec_emb_dim =  config['mp2vec_emb_dim']

with open(os.path.join(DATA_LOC, 'epoch_fileList.json'), 'r') as fh:
    epoch_fileList = json.load(fh)

subDIR_list = sorted(list(epoch_fileList.keys()))
# Parallel(n_jobs = MP.cpu_count())(
#     delayed(exec_training)(subDIR, train_epochs) for subDIR in subDIR_list
# )

for subDIR in subDIR_list:
    exec_training(subDIR, train_epochs)