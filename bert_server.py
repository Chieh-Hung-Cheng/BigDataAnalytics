from bert_serving.client import BertClient
import doc_utils
import numpy as np
import pandas as pd
import os

def bert_server_test():
    bc = BertClient()
    shoes = pd.read_pkl(os.path.join(doc_utils.proc_pth, 'Shoes05211728.pkl'))
    for title in shoes['SalePageTitle']:
        x = bc.encode(title)
        print(bc.encode(title))

if __name__ == '__main__':
    bert_server_test()
