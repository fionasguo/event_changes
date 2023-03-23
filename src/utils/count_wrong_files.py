import pandas as pd
import os
from glob import glob
from tqdm import tqdm

data_dir = '/data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/'
files = glob(data_dir + '*.csv')
cnt = 0

for file in tqdm(files):
    open()

    # df = pd.read_csv(file,lineterminator='\n')
    # # if len(df[df.text.str.startswith('rt ')])>0:
    # if len(df[df.tweetid.isnull()]>0):
    #     print(file)
    #     print()
    #     cnt += 1

print(cnt)
