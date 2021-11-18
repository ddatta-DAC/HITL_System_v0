'''
STATIC DATA FETCHER

# -----------------------------
Fetch the saved html files for each type of chart
'''
# ---------------------
# [Note : set base_path to have the correct relative path
# ---------------------

import os
import sys
import pandas as pd
from glob import glob 
from tqdm import tqdm
import multiprocessing as MP
from joblib import Parallel, delayed


class static_data_fetcher():

    def __init__(self, base_path = './'):
        self.chart_locationDir = {
            'companyNetworkViz': 'companyNetworkViz',
            'stackedComparison': 'stackedComparison',
            'EmbViz_all' : 'EmbViz_all',
            'HSCodeViz' : 'HSCodeViz',
            'TimeSeries' : 'TimeSeries',
            'sankeyDiagram' : 'sankeyDiagram'
        }
        self.base_path = base_path
        self.htmlCache_dir_name = 'htmlCache'
        return 
    '''
    sub_DIR is the epoch
    '''
    def fetch_saved_html(self, PanjivaRecordID, sub_DIR='01_2016'):
        result = {}
        PanjivaRecordID = int(PanjivaRecordID)
        for figure_type, loc in self.chart_locationDir.items():
            _dir = os.path.join(self.base_path, loc, self.htmlCache_dir_name + '_' + sub_DIR)
            # Search
            if figure_type == 'stackedComparison':
                result[figure_type] = {}
                _dir = os.path.join(_dir,  str(PanjivaRecordID) )
                
                files = glob(os.path.join(_dir,'**.html'))
                for f in files:
                    domain = os.path.basename(f).split('.')[0].split('__')[-1]
                    fh = open(f, 'r')
                    result[figure_type][domain] = fh.read()
                    fh.close()
            elif figure_type == 'sankeyDiagram':
                result[figure_type] = {}
                # fetch the 2 types of diagrams.
                files = glob(os.path.join(_dir,'**_{}_**.html'.format(PanjivaRecordID)))
                
                for f in files:
                    _type =  os.path.basename(f).split('.')[0].split('_')[-1] 
                    fh = open(f, 'r')
                    result[figure_type]['Sankey Diagram Type {}'.format(_type)] = fh.read()
                    fh.close()
            elif figure_type == 'EmbViz_all':
                file = sorted(glob(os.path.join(_dir,'**_{}.html'.format(PanjivaRecordID))))[0]
                fh = open(file, 'r')
                result[figure_type] = fh.read()
                fh.close()
            elif figure_type == 'companyNetworkViz':
                file = sorted(glob(os.path.join(_dir,'**_{}_**.html'.format(PanjivaRecordID))))[0]
                fh = open(file, 'r')
                result[figure_type] = fh.read()
                fh.close()
            elif figure_type == 'TimeSeries':
                result[figure_type] = {}
                file1 = sorted(glob(os.path.join(_dir,'{}**Consignee**.html'.format(PanjivaRecordID))))[0]
                file2 = sorted(glob(os.path.join(_dir,'{}**Shipper**.html'.format(PanjivaRecordID))))[0]
                fh = open(file1, 'r')
                result[figure_type]['Consignee'] = fh.read()
                fh.close()
                fh = open(file2, 'r')
                result[figure_type]['Shipper'] = fh.read()
                fh.close()
            elif figure_type == 'HSCodeViz':
                result[figure_type] = {}
                file1 = sorted(glob(os.path.join(_dir,'{}**Consignee**.html'.format(PanjivaRecordID))))[0]
                file2 = sorted(glob(os.path.join(_dir,'{}**Shipper**.html'.format(PanjivaRecordID))))[0]
                fh = open(file1, 'r')
                result[figure_type]['Consignee'] = fh.read()
                fh.close()
                fh = open(file2, 'r')
                result[figure_type]['Shipper'] = fh.read()
                fh.close()
        return result
    
                    
        

'''
# SAMPLE CALL
'''

# obj = static_data_fetcher(base_path='./')
# result = obj.fetch_saved_html(120748461)
# result.keys()