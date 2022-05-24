from bert_serving.client import BertClient
import doc_utils
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
# conda activate tf1
# bert-serving-start -model_dir D:\PycharmProject_D\bda_final\data\chinese_L-12_H-768_A-12 -num_worker=1

def bert_server_test():
    bc = BertClient()
    shoes = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'Shoes05211728.pkl'))

    shoes_array = np.zeros((len(shoes), 768))
    for idx, title in enumerate(tqdm(shoes['SalePageTitle'])):
        vector = bc.encode([title])
        # print(vector)
        shoes_array[idx, :] = vector

    compare = np.load(os.path.join(doc_utils.proc_pth, 'title_bert.npy'))
    pass

def comparision():
    original = np.load(os.path.join(doc_utils.proc_pth, 'Results', 'ShoesTitles', 'title_bert.npy'))
    svr = np.load(os.path.join(doc_utils.proc_pth, 'Results', 'ShoesTitles', 'title_bert_svr.npy'))
    pass

def bert_server_vectorize_phrases():
    bc = BertClient()
    with open(os.path.join(doc_utils.proc_pth, 'Results', 'ItemPhrases', 'phrases.pkl'), 'rb') as f:
        phrases = pickle.load(f)
    pass

    phrase_array = np.zeros((len(phrases), 768))
    for idx, phr in enumerate(tqdm(phrases)):
        vector = bc.encode([phr])
        # print(vector)
        phrase_array[idx, :] = vector

    with open(os.path.join(doc_utils.proc_pth, 'Results', 'ItemPhrases', 'phrase_array_svr.npy'), 'wb') as f:
        np.save(f, phrase_array)

if __name__ == '__main__':
    bert_server_vectorize_phrases()
    #bert_server_test()
