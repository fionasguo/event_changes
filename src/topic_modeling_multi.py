import os
import pickle
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils.preprocessing_topic_modeling import preprocess_tweet


# load changepoints - dict of dicts {concern:{mf:[(timestamp,confidence),(timestamp,confidence),...]}}
with open('../covid_data/changepoints_covid.pkl','rb') as f:
    changepoints = pickle.load(f)
# changepoints = {
#     'all':{'fairness': [
#         (pd.Timestamp('2020-08-26 00:00:00+0000', tz='UTC'), 0.9694291002937403),
#         (pd.Timestamp('2021-01-31 00:00:00+0000', tz='UTC'), 0.8608076939807563),
#         (pd.Timestamp('2020-06-18 00:00:00+0000', tz='UTC'), 0.999991027740641)
#     ]}
# }

#data_dir = '/data/Coronavirus-Tweets/Covid19_Full_Dataset/mf_annotations/'
df = pd.read_csv('/nas/home/siyiguo/event_changes/covid_data/covid_10perc_mf_concern.csv',lineterminator='\n')
df['date'] = pd.to_datetime(df['date'])

for concern,v in changepoints.items():
    for mf,cp_list in v.items():
        for i in range(len(cp_list)):
            cp = cp_list[i]
            print(concern,mf,cp[0])
            fname=concern+'_'+mf+'_'+str(cp[0]).split()[0]
            if os.path.exists('/nas/home/siyiguo/event_changes/covid_topic_model/tm_'+fname): continue

            event_date = cp[0] # pd.Timestamp with tz='UTC'
            start_date = event_date - pd.Timedelta(5, unit='D')
            end_date = event_date + pd.Timedelta(5, unit='D')

            # retrieve the data
            df_tmp = df[(df['date']>=event_date) & (df['date']<end_date) & (df[mf]==1) & (df[concern]==1)]
            df_tmp = df_tmp.sort_values('date')
            print(len(df))
            timestamps = df_tmp['date'].to_list()
            tweets = df_tmp.text.to_list()

            # get global topic model
            print('start topic modeling')
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            topic_model = BERTopic(embedding_model=sentence_model, verbose=False)
            topics, probs = topic_model.fit_transform(tweets)

            topic_model.save('/nas/home/siyiguo/event_changes/covid_topic_model/tm_'+fname, save_embedding_model=False)
            print('global topic model saved')

            # dynamic topic modeling
            topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
            with open('/nas/home/siyiguo/event_changes/covid_topic_model/dtm_'+fname+'.pkl','wb+') as f:
                pickle.dump(topics_over_time,f)
            print('dynamic topic modeling done')
