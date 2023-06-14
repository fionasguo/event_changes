"""
Aggregate emotion detection results on individual tweets to time series
command to run: 
python aggregate_to_ts.py "/data/Coronavirus-Tweets/Covid19_Full_Dataset/mf_annotations" "['political','issue']" 
"""


import os
import sys
from glob import glob
import pickle
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from itertools import chain, combinations


def get_all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def get_comb(l1,l2):
    res = []
    for i1 in l1:
        for i2 in l2:
            res.append(i1+i2)
    return res

def get_comb_bt_lists(ll):
    combs = [(item,) for item in ll[0]]
    for i in range(1,len(ll)):
        combs = get_comb(combs,[(item,) for item in ll[i]])
    return combs

############## filter for/out keywords ##############
def filter(df,text_col,keyword_lst):
    """
    input and output: pd dataframe
    text_col: the column name of tweet texts
    """
    filtered_df = pd.DataFrame()

    for word in keyword_lst:
        tmp = df[df[text_col].str.lower().str.contains(word)]
        filtered_df = pd.concat([filtered_df, tmp],axis=0)

    filtered_df = filtered_df.drop_duplicates()

    return filtered_df

def filter_out(df,text_col,keyword_lst):

    for word in keyword_lst:
        tmp = df[df[text_col].str.lower().str.contains(word)]
        df = df.drop(tmp.index)

    return df

# GF keywords
f = open('/nas/home/siyiguo/event_changes/data/preprocessing/GF_keyword_lst.txt', "r")
GF_keyword_lst = f.readlines()
GF_keyword_lst = [l.replace('\n','') for l in GF_keyword_lst]

# covid - use emily Chen's keywords
f = open('/nas/home/siyiguo/event_changes/data/preprocessing/covid_keyword_lst.txt', "r")
covid_keyword_lst = f.readlines()
covid_keyword_lst = [l.replace('\n','') for l in covid_keyword_lst]

############## files for covariates ##############
#political ideology
users = pd.read_csv('/nas/home/siyiguo/event_changes/political-and-science.csv')
users = pd.concat([users['user'],pd.get_dummies(users.label_x)[['Pro-Science','Conspiracy-Pseudoscience']],pd.get_dummies(users.label_y)[['left','right']]],axis=1)
users.columns = ['screen_name','Pro_Science','Conspiracy_Pseudoscience','Left','Right']

emot_mf_colnames = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
    'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
    'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
    'degradation'
    ]
mf_colnames = [
        'care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 
        'authority', 'subversion', 'purity', 'degradation'
    ]
issue_colnames = [
        'origins', 'lockdowns', 'masking', 'healthcare',
        'education', 'therapeutics', 'vaccines'
    ]

cov2values = {
    'political': ['Left','Right'],
    'science': ['Pro_Science','Conspiracy_Pseudoscience'],
    'issue': issue_colnames
}


############## aggregate ##############
def aggregate_df(data_dir,covariates=[],issue_dir=None,fname=''):
    # iterate thru data folder
    files = glob(data_dir + '*.pkl')
    cnt = 0

    df_tot = pd.DataFrame()

    for file in tqdm(files):
        # read data with MF annotations
        df = pd.read_csv(file, lineterminator='\n')
        # print(len(df))

        # read data with issue annotations if issue is in the covariates
        if 'issue' in covariates:
            issue_file = os.path.join(issue_dir,file.split('/')[-1])
            df_issue = pickle.load(open(issue_file,'rb'))
            # print(len(df_issue))
        
        ## filter for/out keywords
        # df = filter(df,'text',covid_keyword_lst)
        # print(len(df))
        # df = filter_out(df,'text',covid_keyword_lst)
        # print(len(df))
        
        # agg to time series based on the entire population first
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.round('H')
        df_agg = df.groupby('date')[mf_colnames].sum()
        df_agg['count'] = df.groupby('date')['tweetid'].count()
        df_agg[covariates] = 'all'
        # print(len(df_agg))
        df_template = df_agg.copy()

        # agg to t.s. for subpopulations based on covariates
        for cov in get_all_subsets(covariates):
            cov=list(cov)
            # print('cov',cov)
            if not cov: continue
            # deal with covariates
            df_tmp = df.copy()
            if 'political' in cov or 'science' in cov:
                df_tmp = df_tmp.merge(users,on='screen_name',how='inner')
            if 'issue' in cov:
                df_tmp = df_tmp.merge(df_issue[['tweetid','concerns']+issue_colnames],on='tweetid',how='inner')
                df_tmp = df_tmp[df_tmp.concerns.apply(len)>0]
            # print(len(df_tmp))
            # print(get_comb_bt_lists([cov2values[c] for c in cov]))
            for subpopulation in get_comb_bt_lists([cov2values[c] for c in cov]):
                # print(subpopulation)
                query_str = " == 1 & ".join(subpopulation + ('',))[:-3]
                # print(query_str)
                # print(df_tmp.query(query_str))
                tmp_agg = df_tmp.query(query_str).groupby('date')[mf_colnames].sum()
                tmp_agg['count'] = df_tmp.query(query_str).groupby('date')['tweetid'].count()
                for c in covariates:
                    value = list(set(cov2values[c]).intersection(set(subpopulation)))
                    if not value: value = 'all'
                    else: value = value[0]
                    tmp_agg[c] = value
                tmp_agg = tmp_agg.reindex_like(df_template)
                # print(tmp_agg)

                df_agg = pd.concat([df_agg,tmp_agg],axis=0)
                # print(len(df_agg))

        df_tot = pd.concat([df_tot,df_agg],axis=0)
        # print(len(df_tot))

        # df_tot = df_tot.groupby('date')[mf_colnames+['count']].sum()

        df_tot.to_csv(f'~/event_changes/covid_data/agg_{fname}.csv')

if __name__ == '__main__':
    # data_dir = sys.argv[1]
    data_dir = "/data/Coronavirus-Tweets/Covid19_Full_Dataset/mf_annotations/"
    
    
    aggregate_df(data_dir,covariates,issue_dir,'_'.join(covariates))
