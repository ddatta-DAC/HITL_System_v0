import os
import sys
VALID_HSCODE_LIST = []
with open('./../DomainData/valid_HSCode.txt','r') as fh:
    VALID_HSCODE_LIST = fh.readlines()
VALID_HSCODE_LIST = [_.strip('\n') for _ in VALID_HSCODE_LIST]

print(VALID_HSCODE_LIST)

def HSCode_filter_aux(val):
    global VALID_HSCODE_LIST
    vals = val.split(';')
    for val in vals:
        val = str(val[0])
        val = val.replace('.', '')
        val = str(val[:6])
        if val[:2] == '44':
            return val
        elif val in VALID_HSCODE_LIST:
            return val
        else:
            continue

    return None

