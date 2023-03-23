import os
import pandas as pd


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

data_dir = '/nas/home/siyiguo/event_changes/data/LA_tweets_10perc_sample_emot_mf.csv'
emot_mf_colnames = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
    'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
    'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
    'degradation'
]
df = pd.read_csv(data_dir, lineterminator='\n')
print(len(df))
df = filter_out(df,'text',covid_keyword_lst)
print(len(df))
df = filter_out(df,'text',GF_keyword_lst)
print(len(df))
# df = filter_out(df,'text',covid_keyword_lst)
# print(len(df))


df.to_csv('~/event_changes/data/LA_tweets_10perc_sample_emot_mf_wo_covid_GF.csv')
