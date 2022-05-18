import numpy as np
import pandas as pd
import os

import doc_utils

def slaves_for_members():
    slaves = doc_utils.read_slave_csv()
    members = doc_utils.read_member_csv()
    save_pth = os.path.join(doc_utils.proc_pth, 'slaves4members')
    for index, id in enumerate(members['MemberId']):
        tmp = slaves.loc[slaves['MemberId'] == id]
        if len(tmp) != 0:
            tmp.to_csv(os.path.join(save_pth, '{}.csv'.format(index)))



if __name__ == '__main__':
    slaves_for_members()
    pass