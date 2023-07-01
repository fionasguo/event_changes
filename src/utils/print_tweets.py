import pandas as pd

data_dir = '/nas/home/siyiguo/LA_tweets_emot_mf/'

target_events = [('fairness','20200120'),('betrayal','20200120'),
                 ('subversion','20200121'),('surprise','20200404'),
                 ('purity','20200405'),('fairness','20200505'),
                 ('subversion','20200511'),('harm','20200709')]

for e in target_events:
    print(e)
    df = pd.concat([pd.read_csv(data_dir+e[1]+'.csv',lineterminator='\n'),
                    pd.read_csv(data_dir+str(int(e[1])+1)+'.csv',lineterminator='\n'),
                    pd.read_csv(data_dir+str(int(e[1])+2)+'.csv',lineterminator='\n')],axis=0)
    df = df.reset_index(drop=True)

    df = df[df[e[0]]==1]
    print('total number:',len(df))

    for r,row in df.iterrows():
        print(row['text'])
    print('\n\n')



