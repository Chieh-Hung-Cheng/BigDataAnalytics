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
    for year in years:
        df_lst = []
        df_full = []
        for filename in os.listdir(behv_pth):
            print(filename)
            if filename.startswith('91APP_BehaviorData_{}'.format(year)):
                df = pd.read_csv(os.path.join(behv_pth, filename))
                df = df.loc[df['MemberId'].notna()]
                df_lst.append(df)
        df_full = pd.concat(df_lst)
        if save: df_full.to_csv(os.path.join(proc_pth, 'BehaviorMembers_{}.csv').format(year))


def read_behav_csv(year):
    return pd.read_csv(os.path.join(proc_pth, 'BehaviorMembers', 'BehaviorMembers_{}.csv'.format(year)))


def read_member_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_MemberData.csv'))


def read_master_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_OrderData.csv'))


def read_slave_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_OrderSlaveData.csv'))


def read_page_csv():
    return pd.read_csv(os.path.join(dataset_pth, '91APP_SalePageData.csv'))

def trans_datetime(behav_df, save=False, filename='tmp'):
    behav_df['HitDateTime'] = behav_df['HitTime'].apply(lambda x: pd.to_datetime(x, unit='ms')+timedelta(hours=8))
    if save: behav_df.to_csv(os.path.join(proc_pth, '{}.csv'.format(filename)))
    return behav_df
if __name__ == '__main__':
    df = read_behav_csv(2020)
    pass
