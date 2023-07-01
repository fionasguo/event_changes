import pandas as pd
from tqdm import tqdm

N = 20

df = pd.read_csv('/nas/home/siyiguo/event_changes/data/LA_tweets_10perc_sample_emot_mf.csv',lineterminator='\n')

emot_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
mf_cols = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity', 'degradation']
keywords = ['covid','quarantine','coronavirus','kobe','bryant','blacklivesmatter','protest','valentine']


def check_can_drop(df_tmp):
    can_drop = True
    for emotion in emot_cols:
        for i in [0,1]:
            if len(df_tmp[df_tmp[emotion]==i])<N:
                can_drop = False
    for mf in mf_cols:
        for i in [0,1]:
            if len(df_tmp[df_tmp[mf]==i])<N:
                can_drop = False
    for keyword in keywords:
        if len(df[df.text.str.lower().str.contains(keyword)])<N:
            can_drop=False
    return can_drop


df_annot = pd.DataFrame()
for emotion in emot_cols:
    for i in [0,1]:
        df_tmp = df[df[emotion]==i]
        df_tmp = df_tmp.sample(n=N,random_state=33)
        df_annot = pd.concat([df_annot,df_tmp],axis=0)
for mf in mf_cols:
    for i in [0,1]:
        df_tmp = df[df[mf]==i]
        df_tmp = df_tmp.sample(n=N,random_state=33)
        df_annot = pd.concat([df_annot,df_tmp],axis=0)
for keyword in keywords:
    df_tmp = df[df.text.str.lower().str.contains(keyword)]
    df_tmp = df_tmp.sample(n=N,random_state=33)
    df_annot = pd.concat([df_annot,df_tmp],axis=0)

print(len(df_annot))
df_annot = df_annot.drop_duplicates(subset=['text'])
print(len(df_annot))
df_annot.to_csv('/nas/home/siyiguo/event_changes/data/LA_tweets_for_annotation.csv',index=False)
df_backup = df_annot.copy()

for d in tqdm(range(N,N+30)):
    for emotion in emot_cols:
        for i in [0,1]:
            # print(emotion,i,mf,j)
            df_tmp = df_annot.copy()
            tmp = df_tmp[df_tmp[emotion]==i]
            tmp_drop = tmp.sample(n=max(0,len(tmp) - d),random_state=33)
            # print('tweets to drop ',len(tmp_drop))
            if len(tmp_drop)==0: continue
            df_tmp = df_tmp.drop(tmp_drop.index)

            if check_can_drop(df_tmp):
                print('droped for ',emotion,i)
                df_annot=df_tmp.copy()
    for mf in mf_cols:
        for i in [0,1]:
            # print(emotion,i,mf,j)
            df_tmp = df_annot.copy()
            tmp = df_tmp[df_tmp[mf]==i]
            tmp_drop = tmp.sample(n=max(0,len(tmp) - d),random_state=33)
            # print('tweets to drop ',len(tmp_drop))
            if len(tmp_drop)==0: continue
            df_tmp = df_tmp.drop(tmp_drop.index)

            if check_can_drop(df_tmp):
                print('droped for ',emotion,i)
                df_annot=df_tmp.copy()
    print('len of df_annot',len(df_annot))

df_annot.to_csv('/nas/home/siyiguo/event_changes/data/LA_tweets_for_annotation.csv',index=False)