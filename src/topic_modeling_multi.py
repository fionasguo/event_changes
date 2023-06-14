"""
Perform Bertopic on each change point detected
"""

import os
import sys
import pickle
from glob import glob
import pandas as pd

import bertopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils.preprocessing_topic_modeling import preprocess_tweet


if __name__=='__main__':
    changepoint_dir = sys.argv[1]
    data_dir = sys.argv[2]

    # load changepoints - dict of dicts {concern:{mf:[(timestamp,confidence),(timestamp,confidence),...]}}
    # with open('/nas/home/siyiguo/event_changes/covid_data/changepoints_covid_us.pkl','rb') as f:
    with open(changepoint_dir,'rb') as f:
        changepoints = pickle.load(f)
    # changepoints = {
    #     'all':{'fairness': [
    #         (pd.Timestamp('2020-08-26 00:00:00+0000', tz='UTC'), 0.9694291002937403),
    #         (pd.Timestamp('2021-01-31 00:00:00+0000', tz='UTC'), 0.8608076939807563),
    #         (pd.Timestamp('2020-06-18 00:00:00+0000', tz='UTC'), 0.999991027740641)
    #     ]}
    # }

    #data_dir = '/data/Coronavirus-Tweets/Covid19_Full_Dataset/mf_annotations/'
    # df = pd.read_csv('/nas/home/siyiguo/event_changes/covid_data/covid_US_original_tweets_mf_concern.csv',lineterminator='\n',names=colnames)
    df = pd.read_csv(data_dir,lineterminator='\n') #,names=colnames
    df['date'] = pd.to_datetime(df['date'])

    for mf,cp_list in changepoints.items():
        for i in range(len(cp_list)):
            cp = cp_list[i]
            print(mf,cp[0])
            fname=mf+'_'+str(cp[0]).split()[0]
            if os.path.exists('/nas/home/siyiguo/event_changes/LA_topic_models/tm_'+fname): continue
            # if concern=='origins' and mf=='authority' and cp[0]==pd.Timestamp('2021-06-06 00:00:00+00:00',tz='UTC'): continue

            event_date = cp[0] # pd.Timestamp with tz='UTC'
            start_date = event_date - pd.Timedelta(5, unit='D')
            end_date = event_date + pd.Timedelta(5, unit='D')

            # retrieve the data
            df_tmp = df[(df['date']>=event_date) & (df['date']<end_date) & (df[mf]==1)]
            df_tmp = df_tmp.sort_values('date')
            print(len(df_tmp))
            timestamps = df_tmp['date'].to_list()
            tweets = df_tmp.text.to_list()

            # get global topic model
            print('start topic modeling')
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            topic_model = BERTopic(embedding_model=sentence_model, verbose=False)
            topics, probs = topic_model.fit_transform(tweets)

            topic_model.save('/nas/home/siyiguo/event_changes/covid_topic_models/tm_'+fname, save_embedding_model=False)
            print('global topic model saved')

            # dynamic topic modeling
            topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
            with open('/nas/home/siyiguo/event_changes/covid_topic_models/dtm_'+fname+'.pkl','wb+') as f:
                pickle.dump(topics_over_time,f)
            print('dynamic topic modeling done')


    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = bertopic.backend._utils.select_backend(sentence_model)

    tm_files = glob('/nas/home/siyiguo/event_changes/LA_topic_model/tm*')
    ftopics = open('/nas/home/siyiguo/event_changes/data/LA_change_point_topics.txt','w+')

    cnt = 0
    for file in tm_files:
        ftopics.write(file+'\n')
        topic_model = BERTopic.load(file, embedding_model=model)
        ftopics.write('len of 5-day data: '+str(len(topic_model.probabilities_))+'\n')
        ftopics.write('number of topics: '+str(len(topic_model.topic_labels_))+'\n')
        # ftopics.write(topic_model.get_topics())
        for key, value in topic_model.get_topics().items(): 
            if key >=0 and key <= 10:
                ftopics.write('%s:%s\n' % (key, value))
        ftopics.write('\n\n')
        cnt += 1
        # if cnt >= 9: break
    ftopics.close()