import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict, OrderedDict

# --------------
# Input: object_list :: lsit of lists
# each list is an ordered set of objects
# --------------
def borda_count(object_list=None):

    _len_ = len(object_list[0])
    ranks = np.arange(1, _len_ + 1).astype(float).tolist()[::-1]
    master = None
    for _list in object_list:
        _df = pd.DataFrame(data=np.stack([np.reshape(_list, [-1]), np.reshape(ranks, [-1])], axis=1),
                           columns=['ID', 'rank'])
        if master is None:
            master = _df
        else:
            master = master.append(_df, ignore_index=True)
    df = master.groupby('ID').sum().reset_index()
    df = df.sort_values(by='rank', ascending=False)
    _min = min(df['rank'])
    _max = max(df['rank'])
    df['rank'] = df['rank'].apply(lambda x: (x - _min) / (_max - _min))
    return df


def test():
    x1 = np.arange(1, 11).astype(int)
    x2 = np.arange(1, 11).astype(int)
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    result = borda_count([x1, x2])
    return result
