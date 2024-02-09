import logging

import config

import pandas as pd
import numpy as np


def tokenize(sent):
    return sent.split(" ")


class Lang:
    """Represents the vocabulary
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {
            config.PAD: 0,
            config.UNK: 1,
        }
        self.word2count = {}
        self.index2word = {
            0: config.PAD,
            1: config.UNK,
        }
        self.n_words = 2

    def add_sentence(self, sentence):
        assert isinstance(
            sentence, (list, tuple)
        ), "input to add_sentence must be tokenized"
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __add__(self, other):
        """Returns a new Lang object containing the vocabulary from this and
        the other Lang object
        """
        new_lang = Lang(f"{self.name}_{other.name}")

        # Add vocabulary from both Langs
        for word in self.word2count.keys():
            new_lang.add_word(word)
        for word in other.word2count.keys():
            new_lang.add_word(word)

        # Fix the counts on the new one
        for word in new_lang.word2count.keys():
            new_lang.word2count[word] = self.word2count.get(
                word, 0
            ) + other.word2count.get(word, 0)

        return new_lang


def load_google(split, max_len=None):
    """Load the Google Sentence Compression Dataset"""
    logger = logging.getLogger(f"{__name__}.load_compression")
    lang = Lang("compression")

    if split == "train":
        path = config.google_train_path
    elif split == "val":
        path = config.google_dev_path 
    elif split == "test":
        path = config.google_test_path

    logger.info("loading %s from %s" % (split, path))

    dataset=pd.read_parquet(path)
    df=pd.DataFrame(dataset['compression_ratio'])

    array=np.array(dataset['source_tree'])
    array_f=[]
    for i in range(len(array)):
        datacell=array[i]

        array_f.append(list(datacell['node']['form'][1:]))

    array=np.array(dataset['compression_untransformed'])
    array_f1=[]
    for i in range(len(array)):
        datacell=array[i]

        array_f1.append(datacell['text'])
    df['text']=array_f
    df['compressed']=array_f1
    array_f2=[]
    for i in range(len(array)):
        datacell=df['compressed'].iloc[i]

        array_f2.append(tokenize(datacell))
    df['compressed_list']=array_f2
    df=df.drop(['compression_ratio','compressed'], axis=1)
    array_f3=[]
    cell=[]
    for i in range(len(df)):
        cell1=df['text'].iloc[i]
        cell2=df['compressed_list'].iloc[i]
        for word in cell1:
            if word in cell2:
                cell.append(1)
            else:
                cell.append(0)
        array_f3.append(cell)
        cell=[]
    df['labels']=array_f3
    df=df.drop(['compressed_list'], axis=1)
    
    data = []
    sent = []
    mask = []
    for i in range(len(df)):
        if max_len is None or len(df['text'][i]) <= max_len:
            sent.append(df['text'][i])
            lang.add_sentence(df['text'][i])
            mask.append(df['labels'][i])
            data.append([df['text'][i],df['labels'][i]])


    return data, lang
