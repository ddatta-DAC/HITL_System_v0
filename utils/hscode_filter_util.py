import os
import sys
VALID_HSCODE_LIST = []
with open('./../DomainData/valid_HSCode.txt','r') as fh:
    VALID_HSCODE_LIST = fh.readlines()
VALID_HSCODE_LIST = [_.strip('\n') for _ in VALID_HSCODE_LIST]

print(VALID_HSCODE_LIST)

def HSCode_filter_aux(val):
    global VALID_HSCODE_LIST
    val = str(val)
    vals = val.split(';')

    for _val in vals:
        _val = str(_val)
        _val = _val.replace('.', '')
        _val = str(_val[:6])
        if _val[:2] == '44':
            return _val
        elif _val in VALID_HSCODE_LIST:
            return _val
        else:
            continue

    return None

