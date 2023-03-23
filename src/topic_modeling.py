import re
import pandas as pd
from bertopic import BERTopic
import pickle
from sentence_transformers import SentenceTransformer
from utils.preprocessing_topic_modeling import preprocess_tweet


#df = pd.read_csv('/nas/home/siyiguo/event_changes/data/CIA_Tweets_20perc.csv',lineterminator='\n')
#df = pd.read_csv('~/event_changes/covid_data/covid_10perc_sample_200121_210630.csv',lineterminator='\n')
#print(len(df))
# for r,row in df.iterrows():
#     print(row['text'])
# df = df[df.created_at < '2020-08-01 00:00:00+00:00']

# df.text = df.apply(lambda row: re.sub("http", "", row.text).lower(), 1)
# df.text = df.apply(lambda row: re.sub("user", "", row.text).lower(), 1)
# df.text = df.apply(lambda row: re.sub("amp", "", row.text).lower(), 1)
# df.text = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
# df.text = df.apply(lambda row: " ".join([w for w in word_tokenize(row.text) if (w not in stop_words and len(w)<15)]), 1)
# df.text = df.apply(lambda row: re.sub("  ", "", row.text).lower(), 1)
#df['text'] = df['text'].apply(preprocess_tweet)
#df = df[df.text != " "]
#df = df[df.text != ""]
# df['date'] = pd.to_datetime(pd.to_datetime(df['date']).astype(int),unit='ms')
#df['date'] = pd.to_datetime(df['date'])
#print(len(df))
#df.to_csv('/nas/home/siyiguo/event_changes/covid_data/covid_10perc_sample_200121_210630_processed.csv',index=False)

df = pd.read_csv('/nas/home/siyiguo/event_changes/covid_data/covid_5perc_sample_200121_210630_processed.csv',lineterminator='\n')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date']<=pd.Timestamp('2020-06-30').tz_localize('utc')]
print(len(df))
df.to_csv('/nas/home/siyiguo/event_changes/covid_data/covid_5perc_sample_200121_200630_processed.csv',index=False)

timestamps = df['date'].to_list()
tweets = df.text.to_list()

#with open('/nas/home/siyiguo/event_changes/data/LA_tweets_10perc_sample_processed_tweets.txt','w+') as f:
#    for t in tweets:
#        f.write(t)
#        f.write('\n')

#with open('/nas/home/siyiguo/event_changes/data/LA_tweets_10perc_sample_timestamps.txt','w+') as f:
#    for t in timestamps:
#        f.write(t)
#        f.write('\n')

# get global topic model
print('start topic modeling')
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model, verbose=True)
topics, probs = topic_model.fit_transform(tweets)

topic_model.save('covid_5perc_200121_200630_topic_model', save_embedding_model=False)
# with open('CIA_Tweets_20perc_topics.pkl','wb+') as f:
#     pickle.dump(topics,f)
# with open('CIA_Tweets_20perc_topic_probs.pkl','wb+') as f:
#     pickle.dump(probs,f)
print('global topic model saved')

# dynamic topic modeling
topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
with open('covid_5perc_200121_200630_topics_over_time_.pkl','wb+') as f:
    pickle.dump(topics_over_time,f)
print('dynamic topic modeling done')

