VALID_HSCODE_LIST = [
    '9401', '9403', '9201', '9614', '9202',
    '9302', '9304', '0305', '8211', '6602',
    '8201', '9207', '9504', '9205', '9206',
    '9209', '9202'
]


def HSCode_filter_aux(val):
    global VALID_HSCODE_LIST

    val = val.split(';')
    val = str(val[0])
    val = val.replace('.', '')
    val = str(val[:6])
    if val[:2] == '44':
        return val
    elif val[:4] in VALID_HSCODE_LIST:
        return str(val[:6])
    else:
        return None
