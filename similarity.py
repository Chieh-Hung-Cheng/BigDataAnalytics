import numpy as np
import pandas as pd
import doc_utils
from sklearn import cluster
import pickle

import os
import sys

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

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
    sim_mtx = linear_kernel(member_mtx)

    sample = sim_mtx[10000, :]
    sorted_index_array = np.argsort(sample)[::-1]
    sim_members = member_vectors.iloc[sorted_index_array[0:100, 0]]
    sim_behaviors = member_behaviors.loc[member_behaviors['MemberId'].isin(sim_members)]
    pass


def item_similarity_test():
    pages = doc_utils.read_page('proc', ver='05191557')
    item_mtx = np.stack(pages.Vector)
    sim_mtx = linear_kernel(item_mtx)

    sample = sim_mtx[800, :]
    sorted_index_array = np.argsort(sample)[::-1]
    pages.iloc[sorted_index_array[0:100], :]
    pass


if __name__ == '__main__':
    member_similarity_test()