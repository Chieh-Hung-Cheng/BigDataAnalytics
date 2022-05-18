import os
import sys

import numpy as np
import pandas as pd

from datetime import timedelta

base_pth = os.path.dirname(__file__)
data_pth = os.path.join(base_pth, 'data')

raw_pth = os.path.join(data_pth, 'raw')
dataset_pth = os.path.join(raw_pth, '91APP_DataSet_2022')
eland_pth = os.path.join(raw_pth, '91APP_eland')
behv_pth = os.path.join(dataset_pth, '91APP_BehaviorData')

proc_pth = os.path.join(data_pth, 'proc')


def flt_behav(save=True):
    years = ['2018', '2019', '2020', '2021', '2022']
    intrest_actions = ['viewproduct', 'search', 'add', 'checkout', 'purchase']
    intrest_colnames = ['MemberId', 'HitTime', 'Behavior', 'SalePageId', 'UnitPrice', 'Qty', 'TotalSalesAmount',\
                        'TradesGroupCode', 'SearchTerm', 'EventTime']
    for year in years:
        df_lst = []
        df_full = []
        for filename in os.listdir(behv_pth):
            print(filename)
            if filename.startswith('91APP_BehaviorData_{}'.format(year)):
                df = pd.read_csv(os.path.join(behv_pth, filename))[intrest_colnames]
                df = df.loc[df['MemberId'].notna()]
                df = df.loc[df['Behavior'].isin(intrest_actions)]
                df_lst.append(df)
        df_full = pd.concat(df_lst)
        df_full.sort_values(['MemberId', 'HitTime', 'EventTime'], inplace=True)
        if save: df_full.to_csv(os.path.join(proc_pth, 'BehaviorMembersAbbr', 'BehaviorMembers_{}.csv').format(year),\
                                index=False)

def flt_behav_crit(save=True):
    years = ['2018', '2019', '2020', '2021', '2022']
    intrest_actions = ['search', 'add', 'checkout', 'purchase']

    df_lst = []
    for filename in os.listdir(os.path.join(proc_pth, 'BehaviorMembersAbbr')):
        print(filename)
        df = pd.read_csv(os.path.join(proc_pth, 'BehaviorMembersAbbr', filename))
        df = df.loc[df['Behavior'].isin(intrest_actions)]
        df_lst.append(df)
    df_full = pd.concat(df_lst)
    df_full.sort_values(['MemberId', 'HitTime', 'EventTime'], inplace=True)
    df = trans_datetime(df_full)
    if save: df.to_pickle(os.path.join(proc_pth, 'BehaviorMembersCrit.pkl'))

def read_behav_csv(year):
    return pd.read_csv(os.path.join(proc_pth, 'BehaviorMembers', 'BehaviorMembers_{}.csv'.format(year)))


def read_member_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_MemberData.csv'))


def read_master_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_OrderData.csv'))


def read_slave_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_OrderSlaveData.csv'))


def read_page_csv(pth='raw'):
    if pth=='raw': return pd.read_csv(os.path.join(dataset_pth, '91APP_SalePageData.csv'))
    elif pth=='proc': return pd.read_pickle(os.path.join(proc_pth, 'SalePage_Vec.pkl'))


def trans_datetime(behav_df, save=False, filename='tmp'):
    # behav_df['HitDT'] = behav_df['HitTime'].apply(lambda x: pd.to_datetime(x, unit='ms') + timedelta(hours=8))
    # behav_df['EventDT'] = behav_df['EventTime'].apply(lambda x: pd.to_datetime(x, unit='ms') + timedelta(hours=8))
    behav_df['HitDT'] = pd.to_datetime(behav_df['HitTime'], unit='ms') + timedelta(hours=8)
    behav_df['EventDT'] = pd.to_datetime(behav_df['EventTime'], unit='ms') + timedelta(hours=8)
    if save: behav_df.to_csv(os.path.join(proc_pth, '{}.csv'.format(filename)))
    return behav_df

def sort_behv_session():
    df = pd.read_csv(os.path.join(proc_pth, 'BehaviorMembersAbbr', 'BehaviorMembers_2022.csv'))
    df = df.loc[df['Behavior']!='viewproduct']
    df = trans_datetime(df)
    pages = read_page_csv('proc')
    df = pd.merge(df, pages, on='SalePageId', how='left')

    user_groups = df.groupby('MemberId')
    for userid, actions in user_groups:
        if len(actions) < 10: continue
        sessions = actions.groupby('HitTime')
        for hittime, session in sessions:
            pass
    pass

def doc_test():
    df = pd.read_pickle(os.path.join(proc_pth, 'BehaviorMembersCrit.pkl'))
    pages = read_page_csv('proc')
    df = pd.merge(df, pages, on='SalePageId', how='left')
    for memberid, histories in df.groupby('MemberId'):
        pass


if __name__ == '__main__':
    doc_test()
    pass
