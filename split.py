import pandas as pd
import numpy as np
from gensim.models import word2vec
import re
import os
import doc_utils
from tqdm import tqdm
import pickle


def clear_sentence(sentence):
    return re.sub(r'[^\u4e00-\u9fa5]+', '', sentence)


def has_digit(str):
    return bool(re.search(r'\d', str))



def split_sentence_to_phrase(sentence):
    import monpa
    from monpa import utils
    monpa.use_gpu = True
    sentence = clear_sentence(sentence)
    short_sentences = utils.short_sentence(sentence)
    slices = []
    if monpa.use_gpu:
        result_cut_batch = monpa.cut_batch(short_sentences)
        for i in result_cut_batch:
            slices += i
    else:
        for elm in short_sentences:
            slices += monpa.cut(elm)
    return [i.strip(' -') for i in slices if len(i.strip(' -')) >= 2 and not has_digit(i.strip())]


def gen_word2vec_model():
    pages = doc_utils.read_page_csv()
    page_titles = pages['SalePageTitle']
    lst_of_lsts = []
    with open(os.path.join(doc_utils.proc_pth, 'phrases.txt'), 'w') as file:
        for title in tqdm(page_titles):
            lst = split_sentence_to_phrase(title)
            file.write('{}\n'.format(lst))
            lst_of_lsts.append(lst)
    word2vec_model = word2vec.Word2Vec(lst_of_lsts, vector_size=1000, min_count=1, window=5)
    words = list(word2vec_model.wv.index_to_key)
    word_vec = {word: word2vec_model.wv[word] for word in words}
    word2vec_model.save(os.path.join(doc_utils.proc_pth, 'word2vev.model'))
    pages['SpiltPhrases'] = lst_of_lsts
    pages.to_pickle(os.path.join(doc_utils.proc_pth, 'SalePage.pkl'))
    pass

def add_vector_for_items():
    pages = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'SalePage.pkl'))
    model = word2vec.Word2Vec.load(os.path.join(doc_utils.proc_pth, 'word2vec.model'))

    vector_lst = []
    for phrases in pages['SpiltPhrases']:
        vec = np.zeros(1000)
        for phr in phrases:
            vec += model.wv[phr]
        # Normalize
        if len(phrases) !=0: vec /= len(phrases)
        vector_lst.append(vec)
    pages['Vector'] = vector_lst
    pages.to_pickle(os.path.join(doc_utils.proc_pth, 'SalePage_Vec.pkl'))
    pass


if __name__ == '__main__':
    gen = False
    if gen: gen_word2vec_model()
    add_vector_for_items()
    pass