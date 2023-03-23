import numpy as np
import pandas as pd
import json
import os
import datetime

from preprocessing import preprocess_tweet

dir_path = '/lfs1/jeff_data'
start = datetime.date(2020, 1, 1)
end = datetime.date(2020, 12, 31)

for filename in os.listdir(dir_path):
    date_time_obj = datetime.datetime.strptime(
        filename.split('.')[0], '%Y%m%d')
    # print(date_time_obj.date())

    if date_time_obj.date() >= start and date_time_obj.date() <= end:
        tmp = filename.split('.')[0]
        if os.path.exists(f'/nas/home/siyiguo/LA_tweets/{tmp}.csv'): continue

        print(date_time_obj.date())
        # read data
        df = pd.read_json(f'{dir_path}/{filename}', lines=True)
        # filter for english tweets
        df = df[df.lang == 'en']
        df.reset_index(drop=True, inplace=True)
        # get user information
        users = pd.json_normalize(df.user)[[
            'id', 'screen_name', 'verified', 'followers_count',
            'friends_count', 'favourites_count'
        ]]
        users.columns = [
            'user_id', 'user_screen_name', 'user_verified',
            'user_followers_count', 'user_friends_count',
            'user_favourites_count'
        ]
        # get extended tweet if exists
        extended_tweets = pd.json_normalize(df.extended_tweet)
        df.loc[~extended_tweets.full_text.isnull(),
               'text'] = extended_tweets.loc[
                   ~extended_tweets.full_text.isnull(), 'full_text']
        # discard unwanted columns
        df = df[['created_at', 'id', 'text']]
        # add user info
        df = pd.concat([df, users], axis=1)
        # processing
        df['processed'] = df['text'].apply(preprocess_tweet)
        # choose the ones with more than 20 chars
        df['length'] = df.text.str.len()
        df = df[df.length > 20]
        df.drop('length',axis=1,inplace=True)
        print('dataset size: ', len(df))
        # save
        tmp = filename.split('.')[0]
        df.to_csv(f'/nas/home/siyiguo/LA_tweets/{tmp}.csv',index=False)
