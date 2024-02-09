import logging
import re
import config

from transformers import BertModel, BertTokenizer
import torch
train_size=50000
val_size=45000
test_size=4000

def tokenize(sent):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.tokenize(sent)



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {
            config.PAD: 0,
            config.UNK: 1,
            config.NOFIX: 2,
            config.SOS: 3,
            config.EOS: 4,
        }
        self.word2count = {}
        self.index2word = {
            0: config.PAD,
            1: config.UNK,
            2: config.NOFIX,
            3: config.SOS,
            4: config.EOS,
        }
        self.n_words = 5

    def add_sentence(self, sentence):
        assert isinstance(
            sentence, (list, tuple)
        ), "input to add_sentence must be a string"
        
        tokens = ['[CLS]'] + sentence + ['[SEP]']
        for token in tokens:
            self.add_word(token)


    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __add__(self, other):
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
    

def load_qqp(split):
    """Load the QQP from Original""" 

    lang = Lang("qqp")

    if split == "train":
        path = config.qqp_train_path
        size=train_size
    elif split == "val":
        path = config.qqp_dev_path
        size=val_size
    elif split == "test":
        path = config.qqp_test_path
        size=test_size  

    pairs = []
    sachin=0
    with open(path) as handle:

        # skip header
        handle.readline()
    
        for line in handle: 
            parts = line.strip().split(",")

            if len(parts)==6:
    
                sno, qid1, qid2, sent1, sent2, rating = parts
                if rating=="1":
                    if sachin<size:
                        sent1 = tokenize(sent1)
                        sent2 = tokenize(sent2)
                        lang.add_sentence(sent1)
                        lang.add_sentence(sent2)
                        pairs.append([sent1, sent2, rating])
                        sachin=sachin+1
                    else:
                        break
                
            #pairs.append([sent1, sent2])

    # MS makes the vocab for paraphrase the same
    return pairs, lang