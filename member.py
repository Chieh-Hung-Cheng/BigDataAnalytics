import numpy as np
import pandas as pd
import os
import pickle

from tqdm import tqdm
from gensim.models import word2vec

import doc_utils
import split


def slaves_for_members():
    slaves = doc_utils.read_slave_csv()
    members = doc_utils.read_member_csv()
    save_pth = os.path.join(doc_utils.proc_pth, 'slaves4members')
    for index, id in enumerate(members['MemberId']):
        tmp = slaves.loc[slaves['MemberId'] == id]
        if len(tmp) != 0:
            tmp.to_csv(os.path.join(save_pth, '{}.csv'.format(index)))

def process_member_behaviors():
    behavior_crit = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'BehaviorMembersCrit.pkl'))
    pages = doc_utils.read_page('proc', '05191557')
    behavior_crit = pd.merge(behavior_crit, pages, on='SalePageId', how='left')
    model = word2vec.Word2Vec.load(os.path.join(doc_utils.proc_pth, 'word2vev{}.model'.format('05191557')))

    weights = {'purchase': 4.5, 'checkout': 2, 'search': 1.5, 'add': 1}
    memberid_vector_dict = {}
    for memberid, histories in tqdm(behavior_crit.groupby('MemberId')):
        member_vector = np.zeros(1000)
        total_w = 0
        for idx, history in histories.iterrows():
            w = weights[history['Behavior']]
            if history['Behavior'] == 'search':
                search_str = history['SearchTerm']
                str_lst = split.split_sentence_to_phrase(search_str)
                tmp_vec = np.zeros(1000)
                for phr in str_lst:
                    if phr in model.wv:
                        tmp_vec += model.wv[phr]
                    if np.linalg.norm(tmp_vec) != 0: member_vector += (tmp_vec/np.linalg.norm(tmp_vec)) * w
            else:
                member_vector += history['Vector'] * w
        if np.linalg.norm(member_vector) != 0: member_vector /= np.linalg.norm(member_vector)
        memberid_vector_dict[memberid] = member_vector
        pass

    with open(os.path.join(doc_utils.proc_pth, 'MemberVector.pkl'), 'wb') as file:
        pickle.dump(memberid_vector_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    pass

if __name__ == '__main__':
    process_member_behaviors()
    pass