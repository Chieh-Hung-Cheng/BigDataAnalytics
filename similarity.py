import numpy as np
import pandas as pd
import doc_utils
from sklearn import cluster
import pickle

import os
import sys

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

from gensim.models import word2vec

def dict_to_df(save=True):
    # Transform Dict of MemberId <-> Vector to DataFrame form and save
    with open(os.path.join(doc_utils.proc_pth, 'MemberVector.pkl'), 'rb') as pickle_file:
        member_vector_dict = pickle.load(pickle_file)
    df = pd.DataFrame(member_vector_dict.items(), columns=['MemberId', 'Vector'])
    df.to_pickle(os.path.join(doc_utils.proc_pth, 'MemberVectorDF.pkl'))

def member_similarity_test():
    member_vectors = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'MemberVectorDF.pkl'))
    member_behaviors = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'BehaviorMembersCrit.pkl'))
    pages = doc_utils.read_page('proc', ver='05191557')
    member_mtx = np.stack(member_vectors.Vector)
    sample = linear_kernel([member_mtx[100980]], member_mtx)[0]

    # sample = sim_mtx[10000, :]
    sorted_index_array = np.argsort(sample)[::-1]
    sim_members = member_vectors.iloc[sorted_index_array[0:100], 0]
    sim_behaviors = member_behaviors.loc[member_behaviors['MemberId'].isin(sim_members)]
    sim_behv_title = pd.merge(sim_behaviors, pages, on='SalePageId', how='left')
    pass


def item_similarity_test():
    shoes = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'Shoes_Vec05211728.pkl'))
    item_mtx = np.stack(shoes.Vector)
    sim_mtx = linear_kernel(item_mtx)

    sample = sim_mtx[800, :]
    sorted_index_array = np.argsort(sample)[::-1]
    shoes.iloc[sorted_index_array[0:100], :]
    pass

def wordvec_similarity_test():
    model = word2vec.Word2Vec.load(os.path.join(doc_utils.proc_pth, 'word2vec05211728.model'))
    for idx in range(len(model.wv)):
        print('word: {}'.format(model.wv.index_to_key[idx]))
        print('Similar Top 10:{}'.format(model.wv.most_similar(model.wv.index_to_key[idx], 10)))
        pass
    pass

if __name__ == '__main__':
    # dict_to_df()
    item_similarity_test()
    # wordvec_similarity_test()
