import pandas as pd
import numpy as np
from gensim.models import word2vec
import re
import os
import doc_utils
from tqdm import tqdm
import pickle
import datetime

from ckiptagger import WS, POS, NER
ws = WS(os.path.join(doc_utils.data_pth, 'ckip/data'))

'''def clear_sentence(sentence):
    # Perhaps eliminate streaming goods?
    # eliminate x cm?
    x = re.sub(r'Ann.?s', '', sentence, flags=re.IGNORECASE)
    x = re.sub(r'【.*】', '', x)
    x = re.sub(r'全館?.*\$\d*', '', x)# x = re.sub(r'全館?.*\d', '', x)
    x = re.sub(r'(\$|NT|市價|市售|原價|滿|現折)\d*', '', x)
    x = re.sub(r'最高.*折', '', x)
    x = re.sub(r'\d(\d|\.)*cm' ,'', x)
    # x = re.sub(r'\$\d*', '', x)
    x = re.sub(r'\d{4,4}.*G', '', x)
    x = re.sub(r'[^(\w | \u4e00-\u9fa5 | \. | \%)]+', ' ', x)
    x = re.sub(r'\u10e6', '', x)
    return x'''
def clear_sentence(sentence):
    x = re.sub(r'^.*Ann.?s', '', sentence, flags=re.IGNORECASE)
    x = re.sub(r'\(.*\)', '', x)
    x = re.sub(r'[^\u4e00-\u9fa5]+', ' ', x)
    # x = re.sub(r'【.*】', '', x)
    return x

def process_sentence(x):
    # Extract size
    size_pattern = r'\([\u4e00-\u9fa5]*\)$'
    size_info= re.search(size_pattern, x)
    if size_info is not None: size_info = size_info.group(0)
    x = re.sub(size_pattern, '', x)

    # Clear emojis
    x = re.sub(r'[^\u4e00-\u9fa5 | \w | \' | \:| % |\.]+', ' ', x)

    # Extract color
    color_pattern = r'\s([\u4e00-\u9fa5]){1,3}$'
    color_info = re.search(color_pattern, x)
    if color_info is not None: color_info = color_info.group(0)
    x = re.sub(color_pattern, '', x)

    # Eliminate Ann's
    x = re.sub(r'^.*Ann.?s', '', x, flags=re.IGNORECASE)

    # Remove space in front/end
    x = x.strip()
    return x, size_info, color_info

def ckip_split(sentence_lst):
    ws_results = ws(sentence_lst)
    return ws_results


def has_digit(str):
    return bool(re.search(r'\d', str))

def has_word(str):
    return bool(re.search(r'[A-Z | a-z | 0-9]', str))


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
    slices_long_enough = [i.strip(' -') for i in slices if len(i.strip(' -')) >= 2 and not has_digit(i.strip())] #perhaps digits are fine?
    return slices_long_enough


def gen_word2vec_model():
    pages = doc_utils.read_page()
    # page_titles = pages['SalePageTitle']
    shoes = pages.loc[(pages['SalePageTitle'].str.contains('鞋')) | (pages['SalePageTitle'].str.contains('靴'))]
    # behav_members = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'BehaviorMembersCrit.pkl'))
    # search_strings = behav_members[behav_members['Behavior']=='search']['SearchTerm'].dropna()
    lst_of_lsts = []
    timenow_str =  datetime.datetime.now().strftime("%m%d%H%M")
    with open(os.path.join(doc_utils.proc_pth, 'phrases{}.txt'.format(timenow_str)), 'w', encoding="utf-8") as file:
        for title in tqdm(shoes['SalePageTitle']):
            lst = split_sentence_to_phrase(title)
            file.write('{}\n'.format(lst))
            lst_of_lsts.append(lst)

        '''for search_str in tqdm(search_strings):
            # 890,474 searches, estimated 3hrs
            lst = split_sentence_to_phrase(search_str)
            print('{}:\n{}'.format(search_str, lst))
            file.write('{}\n'.format(lst))
            lst_of_lsts.append(lst)'''

    word2vec_model = word2vec.Word2Vec(lst_of_lsts, vector_size=250, min_count=1, window=5)
    word2vec_model.save(os.path.join(doc_utils.proc_pth, \
                                     'word2vec{}.model'.format(timenow_str)))
    shoes['SpiltPhrases'] = lst_of_lsts
    shoes.to_pickle(os.path.join(doc_utils.proc_pth, 'Shoes{}.pkl'.format(timenow_str)))
    pass


def add_vector_for_items():
    shoes = pd.read_pickle(os.path.join(doc_utils.proc_pth, 'Shoes05211728.pkl'))
    model = word2vec.Word2Vec.load(os.path.join(doc_utils.proc_pth, 'word2vec05211728.model'))

    vector_lst = []
    for phrases in shoes['SpiltPhrases']:
        vec = np.zeros(250)
        for phr in phrases:
            vec += model.wv[phr]
        # Normalize
        if len(phrases) != 0: vec /= np.linalg.norm(vec)
        # Append
        vector_lst.append(vec)
    shoes['Vector'] = vector_lst
    shoes.to_pickle(os.path.join(doc_utils.proc_pth, 'Shoes_Vec05211728.pkl'))
    pass

def split_test():
    pages = doc_utils.read_page()
    shoe_mask = (pages['SalePageTitle'].str.contains('鞋')) | (pages['SalePageTitle'].str.contains('靴'))


    shoes = pages.loc[shoe_mask].copy()
    notshoes = pages.loc[~shoe_mask].copy()
    bag_mask = (pages['SalePageTitle'].str.contains('包')) & (~pages['SalePageTitle'].str.contains('直播'))
    exclude_mask = pages['SalePageTitle'].str.contains(r'【.*】.*直播$', regex=True)
    bags = pages.loc[bag_mask&(~shoe_mask)].copy()
    nocategory = pages.loc[(~shoe_mask) & (~bag_mask)]

    phrases_lst = []
    for title in tqdm(shoes['SalePageTitle']):
        phrases_lst.append(split_sentence_to_phrase(title))

    shoes['SplitPhrases'] = phrases_lst

    pass

def split_with_color():
    pages = doc_utils.read_page()
    exclude_mask = pages['SalePageTitle'].str.contains(r'【.*】.*直播$', regex=True)
    items = pages[~exclude_mask]

    processed_lst = []
    size_lst = []
    color_lst = []
    for title in tqdm(items['SalePageTitle']):
        x, sz, cr= process_sentence(title)
        processed_lst.append(x)
        size_lst.append(sz)
        color_lst.append(cr)
    items['ProcessedTitle'] = processed_lst
    items['SizeInfo'] = size_lst
    items['ColorInfo'] = color_lst

    split_phrases = []
    long_enough = []
    for idx, row in tqdm(items.iterrows(), total=items.shape[0]):
        sentence_lst = []
        if isinstance(row['ProcessedTitle'], str): sentence_lst.append(row['ProcessedTitle'])
        if isinstance(row['SizeInfo'], str): sentence_lst.append(row['SizeInfo'])
        splitted = ckip_split(sentence_lst)
        split_phrases.append(splitted)
        slices_long_enough = [i.strip(' -') for i in splitted[0] if len(i.strip(' -')) >= 2 and not has_word(i.strip())]
        if row['ColorInfo'] is not None: slices_long_enough += [row['ColorInfo']]
        long_enough.append(slices_long_enough)
        pass
    items['SplitPhrases'] = split_phrases
    items['LongEnough'] = long_enough
    pass




if __name__ == '__main__':
    split_with_color()
    # gen_word2vec_model()
    # add_vector_for_items()
    pass
