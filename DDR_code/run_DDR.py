import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report





def load_data(data_dir, domain=None, seed=1):
    from preprocessing import preprocess_tweet
    """
    Directly load data from files.
    """

    dataset_dict = {}
    # read data
    df = pd.read_csv(data_dir, lineterminator='\n')
    df = df.sample(frac=1,random_state=seed)
    if domain:
        df = df.loc[df['domain']==domain]

    non_zeros = df[['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
                    'degradation']].any(axis=1)
    df = df[non_zeros]
    # preprocess
    df = df[~df.text.isnull()]
    df['prep_text'] = df.text.apply(preprocess_tweet)
    print('~~~ len of df before removing empty %d' % len(df))
    df = df[df['prep_text'].str.strip().astype(bool)]
    print('~~~ len of df after removing empty %d' % len(df))



    df['majority_vote'] = df[
        ['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
         'degradation']].idxmax(axis=1)
    map_mfs = {'care':0, 'harm': 1, 'fairness': 2,
               'cheating': 3,  'loyalty': 4, 'betrayal': 5,
               'authority': 6, 'subversion': 7, 'purity': 8,
               'degradation': 9}
    df = df.replace({'majority_vote': map_mfs})
    df = df.reset_index()
    return df


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = list(range(10))
  roc_auc_lst = []
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_lst.append(roc_auc)

  return roc_auc_lst


if __name__ == '__main__':
    ''' Fitting simple DDR'''
    import gensim.downloader
    from my_interface import get_loadings, calc_dic_vecs


    # data_dir = "~/Desktop/INCAS/MF_inference/DAMF/data/mf_data_10classes.csv"
    data_dir = "~/Desktop/INCAS/event_changes/LA_data/annotating_emot_mf/LA_tweets_for_annotation_gt.csv"
    domain = None

    model = gensim.downloader.load('word2vec-google-news-300')
    num_features = 300
    dic_vecs = calc_dic_vecs(model, num_features)

    map_mfs = {'care':0, 'harm': 1, 'fairness': 2,
                'cheating': 3,  'loyalty': 4, 'betrayal': 5,
                'authority': 6, 'subversion': 7, 'purity': 8,
                'degradation': 9}

    aucs = []
    f1s = []
    for seed in [1]:
        print('===========================')
        print(seed)
        test_data = load_data(data_dir=data_dir, domain=domain, seed=seed)

        results_df = get_loadings(test_data['prep_text'], model, num_features, dic_vecs)
        map_mfs = {'care.virtue': 0, 'care.vice': 1, 'fairness.virtue': 2,
                'fairness.vice': 3, 'loyalty.virtue': 4, 'loyalty.vice': 5,
                'authority.virtue': 5, 'authority.vice': 7, 'sanctity.virtue': 8,
                'sanctity.vice': 9}

        results_df['final_mf'] = results_df.idxmax(axis=1)
        results_df = results_df.replace({'final_mf': map_mfs})
        print("~~~~~ F1 macro")
        print(domain)

        y_true = test_data.loc[~results_df['final_mf'].isnull()]['majority_vote'].values
        y_pred = results_df.loc[~results_df['final_mf'].isnull()]['final_mf'].astype(int).values

        try:
            roc_auc = roc_auc_score_multiclass(y_true, y_pred)
            report  = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            print('auc',roc_auc)
            print(classification_report(y_true, y_pred))

            aucs.append(roc_auc)
            f1 = []
            for mf,metrics in report.items():
                f1.append(metrics['f1-score'])
                f1s.append(f1)
        except:
            print('ROC AUC cant be calculated')
            pass
    
    print('aucs',np.mean(aucs,axis=0),np.std(aucs,axis=0))
    print('f1s',np.mean(f1s,axis=0),np.std(f1s,axis=0))
    print(len(aucs))

    # t = pd.concat([test_data, results_df], axis=1)

    # t.to_csv(f"ddr_{domain}_10classes.csv")



