import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import json
from pathlib import Path
import yaml
import preprocessUtil

with open('config.yaml', 'r') as fh:
    config = yaml.safe_load(fh)

OP_DIR = os.path.join( config['processedData_DIR'], config['subdir'])

with open(os.path.join(OP_DIR, 'epoch_fileList.json'),'r') as fh:
    epoch_fileList = json.load(fh)

    

for epoch,_dict in epoch_fileList.items():
    train_files = _dict['train']
    test_files = _dict['test']
    Path(os.path.join(OP_DIR,str(epoch))).mkdir(exist_ok=True, parents=True)
    output_subdir = os.path.join(OP_DIR,str(epoch))
    preprocessUtil.set_up_config(
        output_subdir
    )
    preprocessUtil.create_train_test_sets(
        train_files,
        test_files
    )
    break










