import os
from glob import glob
import pandas as pd
import numpy as np

data_dir = '/nas/home/siyiguo/LA_tweets_emot_mf/'
id_dir = '/nas/home/siyiguo/event_changes/data/LA_tweets_10perc_sample.csv'
files = glob(data_dir + '*.csv')
cnt = 0

emot_mf_colnames = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
    'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
    'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
    'degradation'
]

df_tot = pd.read_csv(id_dir,lineterminator='\n')
df_tot = df_tot.drop_duplicates('id')
df_tot[emot_mf_colnames] = np.nan
df_tot = df_tot.set_index('id')
wanted_id = set(df_tot.index)

for file in files:
    df = pd.read_csv(file, lineterminator='\n')
    df.id = df.id.astype(float)
    df = df.drop_duplicates('id')
    df = df.set_index('id')
    df = df[df.index.isin(wanted_id)]

    df_tot = df_tot.fillna(df)
    print(len(df_tot),len(df_tot[df_tot.isnull().any(axis=1)]))

df_tot.to_csv('LA_tweets_10perc_sample_emot_mf.csv')

