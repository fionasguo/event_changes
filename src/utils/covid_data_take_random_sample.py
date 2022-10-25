import pandas as pd
import os
import datetime
import pickle

def filter(tweet_df,text_col,keyword_lst,question_words):
    """
    input and output: pd dataframe
    text_col: the column name of tweet texts
    """
    tweets = tweet_df[text_col]
    filtered_df = pd.DataFrame()

    # keywords
    for word in keyword_lst:
        tmp = tweet_df[tweets.str.lower().str.contains(word)]
        filtered_df = filtered_df.append(tmp)

    # question mark
    tmp = tweet_df[tweets.str.lower().str.contains('?',regex=False)]
    filtered_df = filtered_df.append(tmp)

    # question words
    for word in question_words:
        tmp = tweet_df[(tweets.str.lower().str.contains(': '+word+' ',regex=True)) |
                       (tweets.str.lower().str.contains('^'+word+' ',regex=True)) |
                       (tweets.str.lower().str.contains('. '+word+' ',regex=True)) |
                       (tweets.str.lower().str.contains(', '+word+' ',regex=True))]
        filtered_df = filtered_df.append(tmp)

    filtered_df.drop_duplicates()
    return filtered_df

if __name__ == "__main__":
    proj_path = '/nas/home/siyiguo/covid_causal'
    data_path = '/data/vast/projects/lermanlab'
    text_directory = data_path+'/corona-sent'
    MF_directory = data_path+'/fionaguo/corona-sent-MFs'
    emot_directory = data_path+'/burghardt/corona-sent-MFs_emot'

    # read helper files
    users = pd.read_csv(proj_path+'/user_locations.csv') # screen_name, state, country
    us_users = set(users.loc[users.country=="United States",'screen_name'].values)
    political = pd.read_csv(proj_path+'/anti_pro_sci_user_classification.csv')
    target_users = set(political['user'].values)
    target_users = target_users.initersection(us_users) # screen name

    # uncertainty/unknown
    uncertainty_keywords = ['uncertain','unknown','dont know','not sure','not clear','concern','ambiguity','ambiguous',
                            'anxious','anxiety','skepticism','skeptical','worry','worried','worries','unprdictab',
                            'confusion','confused','doubt','hesitation','hesitated','hesitancy','wonder']
    question_words = ['how','why','what','who','which','whose','where','when','has','have','are','is','do','does','did']

    for folder in os.listdir(emot_directory):
        if folder not in ['2020-01','2020-02','2020-03','2020-04','2020-05']: # in case ./DS_store is in the folder
            continue
        for file in os.listdir(emot_directory+'/'+folder):
            # date
            date_nums = os.path.splitext(file)[0].split('-')
            date = datetime.datetime(int(date_nums[2]),int(date_nums[3]),int(date_nums[4]),int(date_nums[5].split('_')[0]))
            print("starting file: ",date)

            # read files and merge
            df = pd.read_csv(text_directory+'/'+folder+'/'+file.split('_')[0]+'.csv',lineterminator='\n')
            df = df[(df.lang=="en") & (df.screen_name.isin(target_users))]
            df_MF = pd.read_csv(MF_directory+'/'+folder+'/'+file.split('_')[0]+'.csv',lineterminator='\n')
            df_emot = pd.read_csv(emot_directory+'/'+folder+'/'+file,lineterminator='\n')
            df = df.merge(df_MF,left_on='tweetid',right_on='tweetid',how='inner')
            if "tweetid" in df_emot.columns:
                df = df.merge(df_emot,left_on='tweetid',right_on='tweetid',how='inner')
            elif "ID" in df_emot.columns:
                df = df.merge(df_emot,left_on='tweetid',right_on='ID',how='inner')
            else:
                print("emot file column name error, ",file)
                continue

            sub_df = df.sample(frac=0.05,axis=0)
            sub_df.to_csv(proj_path+'/covid_tweets_random_sampled_5perc.csv',header=False,index=False,mode='a')

            # uncertainty/unknown - count question marks and "How", "what"...
            uncertain_tweets = filter(sub_df,'text',uncertainty_keywords,question_words)
            uncertain_tweets.to_csv(proj_path+'/uncertain_tweets.csv',header=False,index=False,mode='a')
