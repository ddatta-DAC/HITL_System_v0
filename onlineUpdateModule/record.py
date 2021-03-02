# ====================================
# Object to store input records
# ====================================
from sklearn.preprocessing import normalize
import glob
import numpy as np
import os
import sklearn
from collections import OrderedDict

class record_class:
    embedding = None
    serialID_to_entityID = None

    @staticmethod
    def __setup_embedding__(embedding_path, serialID_to_entityID, _normalize=True):
        record_class.embedding = {}
        record_class.serialID_to_entityID = serialID_to_entityID
        files = glob.glob(os.path.join(embedding_path, 'mp2v_**_64.npy'))
        for f in sorted(files):
            emb = np.load(f)
            domain = f.split('/')[-1].split('_')[-2]
            if _normalize:
                emb = normalize(emb, axis=1)
           
            record_class.embedding[domain] = emb
        return

    @staticmethod
    def __obtainEntityFeatureInteraction__(domain_1, entity_1, domain_2, entity_2, interaction_type='concat'):
        vec1 = record_class.embedding[domain_1][entity_1]
        vec2 = record_class.embedding[domain_2][entity_2]
        if interaction_type == 'mul':
            return vec1 * vec2
        if interaction_type == 'concat':
            return np.concatenate([vec1, vec2], axis=-1)

    def __init__(self, _record, _label, is_unserialized=True):
        _record = OrderedDict(_record)
        id_col = 'PanjivaRecordID'
        self.id = _record[id_col]
        domains = list(record_class.embedding.keys())
        self.x = []
        self.label = _label

        for d, e in _record.items():
            if d == id_col:
                continue
            if is_unserialized:
                non_serial_id = int(e)
            else:
                non_serial_id = record_class.serialID_to_entityID[int(e)]
            self.x.append(record_class.embedding[d][non_serial_id])
        self.x = np.array(self.x)