import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import pickle
  

rerun_dates = [
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2020-12-21.csv',
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2021-02-01.csv',
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2021-02-02.csv',
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2021-03-16.csv',
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2021-03-17.csv',
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2021-04-24.csv',
    '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/coronavirus-clean-2021-04-25.csv'
]



# colnames
# emot_mf_colnames = [
#     'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
#     'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
#     'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
#     'degradation'
# ]
# mf_colnames = [
#         'care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 
#         'authority', 'subversion', 'purity', 'degradation'
#     ]
info_cols = ['tweetid','userid','screen_name','text','date','tweet_type']
mf_cols = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal',
       'authority', 'subversion', 'purity', 'degradation']
concern_cols = [
        'origins', 'lockdowns', 'masking', 'healthcare',
        'education', 'therapeutics', 'vaccines'
    ]
all_cols = info_cols+mf_cols+concern_cols

# iterate thru data folder
# data_dir = '/nas/home/siyiguo/LA_tweets/'
# old subset of covid data
# mf_dir = "/data/Coronavirus-Tweets/Covid19_Full_Dataset/mf_annotations/"
# concern_dir = "/data/Coronavirus-Tweets/Covid19_Full_Dataset/concerns_map_rt_fix_cleaned/"
mf_dir = '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/'
mf_files = glob(mf_dir + '*.csv')

# political and science
# users = pd.read_csv('/nas/home/siyiguo/vaccine_causal/data/users_political_science_location.csv')
# # users = pd.concat([users['user'],pd.get_dummies(users.label_x)[['Pro-Science','Conspiracy-Pseudoscience']],pd.get_dummies(users.label_y)[['left','right']]],axis=1)
# users.columns = ['screen_name','science','political','location','state']

flog = open('log','w+')

for file in tqdm(rerun_dates):
    print(file)
    df_mf = pd.read_csv(file, lineterminator='\n')
    print(df_mf.shape)
    df_mf = df_mf[all_cols+['concerns']]
    df_mf = df_mf[~df_mf['text'].isnull()]
    df_mf = df_mf.drop_duplicates(subset=['screen_name','tweetid','date'])
    print('len of df_mf',df_mf.shape)

    # sample
    df_mf = df_mf.sample(frac=0.5, random_state=33)
    print('len sampled',len(df_mf))
    print('unique tweet types',len(pd.unique(df_mf.tweet_type)))
    df_mf = df_mf[df_mf.tweet_type=='original']
    print('len of original tweets',len(df_mf))

    df_mf.to_csv('~/event_changes/covid_data/rerun_covid_10perc_mf_concern.csv', header=False, index=False, mode='a+')

    # flog.write(file+'\n')

print(df_mf.columns)


