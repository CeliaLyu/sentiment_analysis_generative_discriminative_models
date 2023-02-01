import os
import functools
import sys
import numpy as np
import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from collections import Counter

import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


PAD_TOKEN = '<pad>'; PAD_INDEX = 0
UNK_TOKEN = '<unk>'; UNK_INDEX = 1


def load_imdb(base_csv):
    """
    Load the IMDB dataset
    :param base_csv: the path of the dataset file.
    :return: train and test set.
    """
    df = pd.read_csv(base_csv, engine='python')
    df.head()
    reviews, sentiments = df['review'].values, df['sentiment'].values
    
    # we are adding an end of sentence word (eos) at the end of the sentence to help
    #    us know when the sentence ends when generating a sentence
    reviews = [i + " eos" for i in reviews]
    
    # remove useless characters '<br />'
    for i, element in enumerate(reviews):
        reviews[i] = element.replace('<br />', ' ')
    
    return reviews, sentiments


def get_train_test_split(base_csv, test_size):
    reviews, sentiments = load_imdb(base_csv)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=test_size, stratify=sentiments, random_state=233)
    print(f'length of train data is {len(x_train)}')
    print(f'length of test data is {len(x_test)}')

    return x_train, x_test, y_train, y_test