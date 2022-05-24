import pandas as pd
import numpy as np
import pickle

import os
import doc_utils
item_pth = os.path.join(doc_utils.proc_pth, 'Results', 'ItemPhrases')

def test_items():
    item_df = pd.read_pickle(os.path.join(item_pth, 'items.pkl'))

    with open(os.path.join(item_pth, 'phrases.pkl'), 'rb') as f:
        phrase_lst = pickle.load(f)

    phrase_array = np.load(os.path.join(item_pth, 'phrase_array.npy'))
    pass

def test():
    with open(os.path.join(item_pth, 'dfnew.pkl'), 'rb') as f:
        df = pickle.load(f)
    for key, val in wordvec_dict.items():
        print(key, val)
    pass

if __name__ == '__main__':
    test()