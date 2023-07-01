"""
Perform Bertopic on each change point detected
"""

import os
import sys
import pickle
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

import bertopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

import nltk
from nltk import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from utils.preprocessing_topic_modeling import preprocess_tweet

def perform_bertopic(tweets, fname, ftopics):
    # get global topic model
    print('start topic modeling')
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=sentence_model, verbose=False)
    _,_ = topic_model.fit_transform(tweets)

    topic_model.save('/nas/home/siyiguo/event_changes/LA_topic_models/tm_'+fname, save_embedding_model=False)
    print('global topic model saved')

    # # dynamic topic modeling
    # topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
    # with open('/nas/home/siyiguo/event_changes/LA_topic_models/dtm_'+fname+'.pkl','wb+') as f:
    #     pickle.dump(topics_over_time,f)
    # print('dynamic topic modeling done')

    ftopics.write('len of data: '+str(len(topic_model.probabilities_))+'\n')
    ftopics.write('number of topics: '+str(len(topic_model.topic_labels_))+'\n')
    # ftopics.write(topic_model.get_topics())
    for key, value in topic_model.get_topics().items(): 
        if key >=0 and key <= 10:
            ftopics.write('%s:%s\n' % (key, [i[0] for i in value if i[0] not in stop_words]))
    ftopics.write('\n')

    return topic_model



if __name__=='__main__':
    changepoint_dir = '/nas/home/siyiguo/event_changes/data/changepoints_LA.pkl' # sys.argv[1]
    data_dir = '/nas/home/siyiguo/event_changes/data/LA_tweets_10perc_sample_emot_mf.csv' # sys.argv[2]
    ftopics = open('/nas/home/siyiguo/event_changes/data/LA_change_point_topics.txt','w+')

    # load changepoints - dict of dicts {concern:{mf:[(timestamp,confidence),(timestamp,confidence),...]}}
    # with open('/nas/home/siyiguo/event_changes/covid_data/changepoints_covid_us.pkl','rb') as f:
    with open(changepoint_dir,'rb') as f:
        changepoints = pickle.load(f)
    
    # add some change points not detected by the algorithm but are very obvious when manually checked
    # changepoints['fear'].append((pd.Timestamp('2020-01-20 00:00:00+0000', tz='UTC'), 0))
    # changepoints['fear'].append((pd.Timestamp('2020-04-03 00:00:00+0000', tz='UTC'), 0))
    # changepoints['love'].append((pd.Timestamp('2020-04-12 00:00:00+0000', tz='UTC'), 0))
    
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
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['text'] = df['text'].apply(preprocess_tweet)
    print('total len of tweet data', len(df))
    print('\n')

    cnt = 0
    for mf,cp_list in changepoints.items():
        for i in range(len(cp_list)):
            df_tmp, df_tmp_bf, df_tmp_af = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

            cp = cp_list[i]
            fname=mf+'_'+str(cp[0]).split()[0]
            print(fname)
            # if os.path.exists('/nas/home/siyiguo/event_changes/LA_topic_models/tm_'+fname+'_all'): continue
            # if concern=='origins' and mf=='authority' and cp[0]==pd.Timestamp('2021-06-06 00:00:00+00:00',tz='UTC'): continue

            event_date = cp[0] # pd.Timestamp with tz='UTC'
            start_date = event_date - pd.Timedelta(6, unit='D')
            end_date = event_date + pd.Timedelta(5, unit='D')

            ftopics.write(fname+'\n')

            # data - 7 days before to 5 days after
            ftopics.write('all data (4 days bf to 3 days af):'+'\n')
            print('total len of tweet data', len(df))
            df_tmp = df[(df['created_at']>=(event_date - pd.Timedelta(5, unit='D'))) & (df['created_at']<(event_date + pd.Timedelta(7, unit='D'))) & (df[mf]==1)]
            print(f"min date={df_tmp['created_at'].min()}, max date={df_tmp['created_at'].max()}, len={len(df_tmp['created_at'])}")
            tweets = df_tmp.text.to_list()
            timestamps = df_tmp['created_at'].to_list()
            topic_model = perform_bertopic(tweets,fname+'_all',ftopics)
            #ftopics.write('\n\n')

            def get_topic_label(idx):
                return topic_model.topic_labels_[idx]
            df_tmp['topic_idx'] = topic_model.topics_
            n_topics = len(pd.unique(df_tmp['topic_idx']))
            df_tmp['topic'] = df_tmp['topic_idx'].apply(get_topic_label)
            df_tmp['created_at'] = df_tmp['created_at'].dt.round('D')
            df_topic_freq = df_tmp.groupby(['topic_idx','created_at'])['id'].count()
            df_topic_freq = df_topic_freq.reset_index()
            df_topic_freq.columns = ['topic_idx','created_at','Frequency']
            print(f"df_topic_freq: min date={df_topic_freq['created_at'].min()}, max date={df_topic_freq['created_at'].max()}")
            df_topic_freq.to_csv('/nas/home/siyiguo/event_changes/data/tmp_topics/'+fname+'.csv')
            plt.figure()
            colors = plt.cm.tab10.colors
            ax = plt.subplot()
            for t,c in zip(range(min(10,n_topics)),colors):
                try:
                    tmp = df_topic_freq[df_topic_freq.topic_idx==t].sort_values('created_at')
                    name = topic_model.topic_labels_[t]
                    name = "_".join(name.split('_')[1:4])
                    print(f'plot {name}')
                    tmp.plot('created_at','Frequency',kind='line',label=name,ax=ax,color=c) #,figsize=[8,3]
                except:
                    pass
            plt.xlabel('')
            plt.ylabel('Tweet Frequency')
            plt.title(fname)
            #plt.yscale('log')
            # plt.ylim((-10,170))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
            plt.savefig('/nas/home/siyiguo/event_changes/LA_topic_models/figs/'+fname, bbox_inches='tight')
            plt.close('all')

            # before data
            ftopics.write('before data (4 days bf to 1 days bf):'+'\n')
            print('total len of tweet data', len(df))
            df_tmp_bf = df[(df['created_at']>=start_date) & (df['created_at']<(event_date-pd.Timedelta(1, unit='D'))) & (df[mf]==1)]
            print(f"min date={df_tmp_bf['created_at'].min()}, max date={df_tmp_bf['created_at'].max()}, len={len(df_tmp_bf['created_at'])}")
            perform_bertopic(df_tmp_bf.text.to_list(),fname+'_before',ftopics)
            #ftopics.write('\n\n')

            # after data
            ftopics.write('after data (event to 3 days af):'+'\n')
            print('total len of tweet data', len(df))
            df_tmp_af = df[(df['created_at']>=event_date) & (df['created_at']<end_date) & (df[mf]==1)]
            print(f"min date={df_tmp_af['created_at'].min()}, max date={df_tmp_af['created_at'].max()}, len={len(df_tmp_af['created_at'])}")
            perform_bertopic(df_tmp_af.text.to_list(),fname+'_after',ftopics)
            ftopics.write('\n\n')

            cnt += 1
            # if cnt >= 1: break
        # if cnt >= 1: break

    # sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    # model = bertopic.backend._utils.select_backend(sentence_model)

    # tm_files = glob('/nas/home/siyiguo/event_changes/LA_topic_models/tm*')
    

    # cnt = 0
    # for file in tm_files:
    #     ftopics.write(file+'\n')
    #     topic_model = BERTopic.load(file, embedding_model=model)
    #     ftopics.write('len of 7-day data: '+str(len(topic_model.probabilities_))+'\n')
    #     ftopics.write('number of topics: '+str(len(topic_model.topic_labels_))+'\n')
    #     # ftopics.write(topic_model.get_topics())
    #     for key, value in topic_model.get_topics().items(): 
    #         if key >=0 and key <= 10:
    #             ftopics.write('%s:%s\n' % (key, [(i[0],round(i[1],3)) for i in value]))
    #     ftopics.write('\n\n')
    #     cnt += 1
    #     # if cnt >= 9: break
    ftopics.close()