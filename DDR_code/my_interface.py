from __future__ import division

import logging
import time as tm

import numpy as np
import pandas as pd
from cosine_similarity import cos_similarity
from simple_progress_bar import update_progress

datetime = tm.localtime()
date = "{0:}-{1:}-{2:}".format(
    datetime.tm_mon, datetime.tm_mday, datetime.tm_year
)
time = "{0:}:{1:}:{2:}".format(
    datetime.tm_hour, datetime.tm_min, datetime.tm_sec
)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def make_agg_vec(words, model, num_features, model_word_set, filter_out=[]):
    """Create aggregate representation of list of words"""

    feature_vec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    for word in words:
        if word not in filter_out:
            if word in model_word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, model[word])
            # else:
            #     print(f"{word} not in vocabulary")
    # print(nwords)
    avg_feature_vec = feature_vec / nwords

    return avg_feature_vec


def calc_dic_vecs(model, num_features, filter_out=[]):
    """

    :param dic_terms: Dictionary where keys are dimension names and values are terms.
    :param model: word2vec model
    :param num_features: Number of dimensions in word2vec model
    :param model_word_set: Set of unique words in word2vec model
    :param filter_out: Words to exclude from aggregation
    :return: A dictionary where keys are dimension names and values the latent semantic space
     representing that dimension. len(values) will equal num_features.
    """
    mfs_dict = {}
    num_to_mf = {}
    with open('./mfd2.0.dic', 'r') as mfd2:
        reading_keys = False
        for line in mfd2:
            line = line.strip()
            if line == '%' and not reading_keys:
                reading_keys = True
                continue
            if line == '%' and reading_keys:
                reading_keys = False
                print(mfs_dict)
                continue
            if reading_keys:
                num, mf = line.split()
                print(num, mf)
                num_to_mf[num] = mf
                mfs_dict[mf] = []
            else:
                try:
                    word, num = line.split()
                    mf = num_to_mf[num]
                    mfs_dict[mf].append(word)
                except:
                    print(line)
                    print(num_to_mf[num])

    print(mfs_dict.keys())
    agg_dic_vecs = {}
    for mf in mfs_dict.keys():
        agg_dic_vecs[mf] = make_agg_vec(
            mfs_dict[mf],
            model=model,
            num_features=num_features,
            model_word_set=model.key_to_index.keys(),
            filter_out=filter_out,
        )

    return agg_dic_vecs


def get_loadings(docs, model, num_features, dic_vecs):
    """

    :param agg_doc_vecs_path: Path to distributed representations of documents
    :param agg_dic_vecs_path: Path to distributed representations of dictionaries
    :param out_path: Path to write to
    :param num_features: Number of dimensions in distributed representations
    :param delimiter: Delimiter to use
    :return:
    """
    """Get loadings between each document vector in agg-doc_vecs_path and each dictionary dimension in
    agg_dic_vecs_path"""
    # n_docs = float(file_len(agg_doc_vecs_path))
    prog_counter = 0
    counter = 0
    mfs_list = dic_vecs.keys()
    # nan_counter = {"ID": [], "count": 0}
    #######

    # columns = ["ID"] + list(dic_vecs.keys())

    results = []

    for doc in docs:
        prog_counter += 1
        counter += 1
        words = doc.split()
        model_word_set = model.key_to_index.keys()  # model's vocab
        doc_vec = make_agg_vec(words, model, num_features, model_word_set, filter_out=[])
        doc_mf_sims = []
        for mf in mfs_list:
            dic_similarity = cos_similarity(doc_vec, dic_vecs[mf])
            doc_mf_sims.append(dic_similarity)

        results.append(doc_mf_sims)
        if prog_counter >= 0.01 * len(docs):
            prog_counter = 0
            update_progress(counter / (len(docs) - 1))

    out_df = pd.DataFrame(results, columns=mfs_list)
    return out_df
