from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math


ALPHA = 0.3


def get_files():
    path = '/Users/line/axial-coding-master/data/'
    folders = list(os.walk(path))
    files = folders[0][2]
    files.remove('.DS_Store')

    for filename in files:
        file = open(path + filename, 'r')
        text = file.read().strip()
        file.close()
    return files


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!#$%&()*+-./:;<=>?@[\]^_`{|}~"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)       #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data)                 #needed again as we need to stem the words
    data = remove_punctuation(data)       #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data)        #needed again as num2word is giving stop words 101 - one hundred and one
    return data


def extract_text(path, files):
    processed_text = []

    for filename in files:
        file = open(path + filename, 'r', encoding="utf8", errors='ignore')
        text = file.read().strip()
        file.close()

        processed_text.append(word_tokenize(str(preprocess(text))))

    return processed_text


def compute_df(files, processed_text):
    df = {}

    for i in range(len(files)):
        tokens = processed_text[i]
        for w in tokens:
            try:
                df[w].add(i)
            except:
                df[w] = {i}

    for i in df:
        df[i] = len(df[i])

    return df


def doc_freq(word, df_dict):
    c = 0
    try:
        c = df_dict[word]
    except:
        pass
    return c


def compute_tf_idf(files, processed_text, df_dict):
    doc = 0
    n = len(files)

    tf_idf = {}
    for i in range(n):
        tokens = processed_text[i]

        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token, df_dict)
            idf = np.log((n + 1) / (df + 1))

            tf_idf[doc, token] = tf * idf

        doc += 1

    return tf_idf


def matching_score(k, query, tf_idf):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("Matching Score")
    print("\nQuery:", query)
    print("")
    print(tokens)

    query_weights = {}

    for key in tf_idf:

        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")

    l = []
    for i in query_weights[:10]:
        l.append(i[0])

    print(l)




