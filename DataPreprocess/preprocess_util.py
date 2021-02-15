import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from glob import glob
import yaml
import os
import json

def setup_timeEpochs():
    with open('config.yaml','r') as fh:
        config = yaml.safe_load(fh)
    SRC_DATA_LOC = config['SRC_DATA_LOC']
    files = glob.glob(os.path.join(SRC_DATA_LOC,'**.csv'))
    file_dict = {}
    for file in files:
        # parse
        fname = os.path.split(file)[-1]
        name = fname.split('.')[0]
        _path_ = os.path.join(os.path.split(file)[:-1])
        month, year = int(name.split('_')[-2]),  int(name.split('_')[-1])

        prefix = '_'.join(name.split('_')[:-2])
        d_orig = date(year, month, 1) + relativedelta(months=0)
        epoch_key = str(d_orig.month).zfill(2) + '_' + str(d_orig.year)
        train = []

        for m in range(-int(config['train_period']),0,-1):
            d = date(year, month, 1) + relativedelta(months=-m)
            target_fname = os.path.join(_path_, prefix + '_' + str(d.month) + '_' + str(d.year) + '.csv')
            train.append(target_fname)
            if not os.path.exists(target_fname):
                print('Check file!')
        test = []
        for m in range(int(config['test_period'])):
            d = date(year, month, 1) + relativedelta(months=m)
            target_fname = os.path.join(_path_, prefix + '_' + str(d.month) + '_' + str(d.year) + '.csv')
            test.append(target_fname)
            if not os.path.exists(target_fname):
                print('Check file!')
        file_dict[epoch_key] = {}
        file_dict[epoch_key]['train'] = train
        file_dict[epoch_key]['test'] = test

    with open(config['epoch_fileList'], "w") as outfile:
        json.dump(file_dict, outfile)
    return