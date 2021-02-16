import sys
import os

sys.path.append('./..')
sys.path.append('./../..')
import pandas as pd
import json
from pandarallel import pandarallel
pandarallel.initialize()
from dgl.data.utils import save_graphs
import yaml
from pathlib import Path
from operator import itemgetter
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from dgl.data.utils import load_graphs
import pandas as pd
import numpy as np
from torch import FloatTensor as FT
from torch import LongTensor as LT
import multiprocessing as mp
import dgl

cpu_count = mp.cpu_count()
print('CPU count', cpu_count)

# -----------------------------------------------------------
# ----------------------------GLOBALS-------------------------
# -----------------------------------------------------------
MODEL_SAVE_DATA_LOC = None
DIR = None
MODEL_SAVE_DATA_LOC = None

# -----------------------------------------------------------
# Read in data.
# data created by  convertRecord_toGraphData
# -----------------------------------------------------------
def read_graph_data(subDIR):
    global DATA_LOC
    loc = os.path.join(DATA_LOC, subDIR)

    fname_e = 'edges.csv'
    fname_n = 'nodes.csv'

    df_e = pd.read_csv(os.path.join(loc, fname_e), low_memory=False, index_col=None)
    df_n = pd.read_csv(os.path.join(loc, fname_n), low_memory=False, index_col=None)

    # ----------------------------------------
    # replace the node id by synthetic id
    # ----------------------------------------

    print('Types of edges', set(df_e['e_type']))
    edge_weights = {}
    graph_data = {}

    for et in set(df_e['e_type']):
        s, t = et.split('_')
        et_R = '_'.join([t, s])
        tmp = df_e.loc[df_e['e_type'] == et]
        n1 = tmp['source'].values.tolist()
        n2 = tmp['target'].values.tolist()
        weights = tmp['weight'].values.tolist()
        _list = []
        _list_R = []

        print(et, et_R)
        for i, j in zip(n1, n2):
            _list.append((i, j))
            _list_R.append((j, i))
        graph_data[(s, et, t)] = _list
        graph_data[(t, et_R, s)] = _list_R

        edge_weights[et] = weights
        edge_weights[et_R] = weights

    return graph_data, edge_weights


class loss_callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


# -----------------------------------------------------------
# Metapath 2 vec
# -----------------------------------------------------------

def mp2vec(random_walks, epochs=100):
    cpu_count = mp.cpu_count()

    model = Word2Vec(
        random_walks,
        size=128,
        window=3,
        negative=10,
        hs=1,
        min_count=1,
        iter=epochs,
        compute_loss=True,
        null_word='-1',
        callbacks=[loss_callback()]
    )
    return model


# -----------------------------------------------------------
# Get metapaths
# -----------------------------------------------------------
def get_mp_list(graph_obj, multiplier=2):
    with open('metapaths.txt', 'r') as fh:
        mp_list = fh.readlines()

    mp_edges = []
    mp_list = [_.strip('\n') for _ in mp_list]
    mp_list = [_.strip(' ') for _ in mp_list]
    for mp in mp_list:
        mp = mp.split(',')
        _mp = list(mp)
        _mp.reverse()
        mp = mp + _mp[1:]
        e_list = []
        for i in range(len(mp) - 1):
            e = mp[i] + '_' + mp[i + 1]
            e_list.append(e)
            if e not in graph_obj.etypes:
                print('ERROR!!')
        mp_edges.append(e_list * multiplier)
    return mp_edges


# -----------------------
# Note :
#  1. DGL can't process serialized ids
#  2. RW needs to have node types preficed for w2v mocel
# -----------------------
def get_RW_list(graph_obj, metapaths):
    start_node_types = [mp[0].split('_')[0] for mp in metapaths]
    RW_list = []

    node_typeID2typename = {}
    for e in enumerate(graph_obj.ntypes):
        node_typeID2typename[e[0]] = e[1]

    def add_prefix(prefix, val):
        return prefix + '_' + str(val)

    for ntype, mp in zip(start_node_types, metapaths):
        #         print(ntype, graph_obj.nodes(ntype).shape)
        RW_mp = dgl.sampling.random_walk(
            graph_obj,
            metapath=mp,
            nodes=graph_obj.nodes(ntype),
            prob='weight'
        )
        #         print(ntype, mp)
        _random_walks = RW_mp[0].data.numpy()
        #         print(_random_walks[0])

        pattern = RW_mp[1].data.numpy().tolist()
        pattern = [node_typeID2typename[_] for _ in pattern]
        #         print(' > ', pattern)
        vectorized_func = np.vectorize(add_prefix)
        _random_walks = vectorized_func(pattern, _random_walks)

        RW_list.extend(_random_walks.tolist())
    return RW_list


# -----------------------------------------------------------
# Get and save the vectors for each node
# -----------------------------------------------------------

def extract_feature_vectors(w2v_model):
    vectors_dict = {}

    for token, vector in w2v_model.wv.vocab.items():
        try:
            _type, _id = token.split('_')
        except:
            print(' >> ', token, type(token))
            continue
        _id = int(_id)
        if _id < 0: continue
        if _type not in vectors_dict.keys():
            vectors_dict[_type] = {}
        vectors_dict[_type][_id] = w2v_model.wv[token]

    return vectors_dict


def save_vectors(node_vectors):
    global DIR
    global MODEL_SAVE_DATA_LOC
    for n_type, _dict in node_vectors.items():
        # sort the vectors by id
        arr_vec = []

        arr_vec = [_[1] for _ in sorted(_dict.items(), key=itemgetter(0))]
        #         for n_id in sorted(_dict.keys()):
        #             arr_vec.append(_dict[n_id])
        arr_vec = np.array(arr_vec)
        fname = 'mp2v_{}.npy'.format(n_type)
        fname = os.path.join(MODEL_SAVE_DATA_LOC, fname)
        np.save(fname, arr_vec)

    return

    edge_weights[et] = weights
    edge_weights[et_R] = weights

    return graph_data, edge_weights


def main(
        subDIR
):
    global matapath2vec_epochs
    global MODEL_SAVE_DATA_LOC

    MODEL_SAVE_DATA_LOC = os.path.join('saved_model_data', subDIR)
    path_obj = Path(MODEL_SAVE_DATA_LOC)
    path_obj.mkdir(exist_ok=True, parents=True)

    graph_data, edge_weights = read_graph_data(subDIR)
    graph_obj = dgl.heterograph(graph_data)
    print('Node types, edge types', graph_obj.ntypes, graph_obj.etypes)
    print('Graph ::', graph_obj)
    for e_type in edge_weights.keys():
        graph_obj[e_type].edata['weight'] = FT(edge_weights[e_type])

    print('Node types, edge types', graph_obj.ntypes, graph_obj.etypes)
    print('Graph ::', graph_obj)

    print('Graph dgl device', graph_obj.device)
    metapaths = get_mp_list(graph_obj, multiplier=5)
    random_walks = get_RW_list(graph_obj, metapaths)
    mp2vec_model = mp2vec(random_walks, epochs=matapath2vec_epochs)
    mp2vec_model.save(os.path.join(MODEL_SAVE_DATA_LOC, "mp2vec.model"))
    node_vectors = extract_feature_vectors(mp2vec_model)
    save_vectors(node_vectors)
    return


# ============================================= #


with open('config.yaml', 'r') as fh:
    config = yaml.safe_load(fh)
DATA_LOC = config['DATA_LOC']
matapath2vec_epochs = int(config['mp2v_epochs'])
with open(os.path.join(DATA_LOC, 'epoch_fileList.json'), 'r') as fh:
    epoch_fileList = json.load(fh)

subDIR_list = list(epoch_fileList.keys())
for subDIR in subDIR_list:
    main(subDIR)
