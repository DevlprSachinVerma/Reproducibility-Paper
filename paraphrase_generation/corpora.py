import logging
import re
import config

train_size=50000
val_size=45000
test_size=4000

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


def load_wiki(split):
    """Load the Wiki from PAWs"""
    logger = logging.getLogger(f"{__name__}.load_wiki")
    lang = Lang("wiki")

    if split == "train":
        path = config.wiki_train_path
    elif split == "val":
        path = config.wiki_dev_path
    elif split == "test":
        path = config.wiki_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    with open(path) as handle:

        # skip header
        handle.readline()

        for line in handle:
            _, sent1, sent2, rating = line.strip().split("\t")
            if rating == "0":
                continue
            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    # MS makes the vocab for paraphrase the same
    return pairs, lang


def load_qqp_paws(split):
    """Load the QQP from PAWs"""
    logger = logging.getLogger(f"{__name__}.load_qqp_paws")
    lang = Lang("qqp_paws")

    if split == "train":
        path = config.qqp_paws_train_path
        
    elif split == "val":
        path = config.qqp_paws_dev_path
      
    elif split == "test":
        path = config.qqp_paws_test_path
       

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    with open(path) as handle:

        # skip header
        handle.readline()

        for line in handle:
            _, sent1, sent2, rating = line.strip().split("\t")
            if rating == "0":
                continue
            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    # MS makes the vocab for paraphrase the same
    return pairs, lang

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


def load_qqp_kag(split):
    """Load the QQP from Kaggle""" #not original right now, expriemnting with kaggle 100K, 3K, 30K split
    logger = logging.getLogger(f"{__name__}.load_qqp_kag")
    lang = Lang("qqp_kag")

    if split == "train":
        path = config.qqp_kag_train_path
    elif split == "val":
        path = config.qqp_kag_dev_path
    elif split == "test":
        path = config.qqp_kag_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    with open(path) as handle:

        # skip header
        handle.readline()

        for line in handle: #when reading the kag version we do not have 4 fields, but rather 3
            rating, sent1, sent2 = line.strip().split("\t")
            if rating == "0":
                continue
            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    # MS makes the vocab for paraphrase the same
    return pairs, lang


def load_msrpc(split):
    """Load the Microsoft Research Paraphrase Corpus (MSRPC)"""
    logger = logging.getLogger(f"{__name__}.load_msrpc")
    lang = Lang("msrpc")

    if split == "train":
        path = config.msrpc_train_path
    elif split == "val":
        path = config.msrpc_dev_path
    elif split == "test":
        path = config.msrpc_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    with open(path) as handle:

        # skip header
        handle.readline()

        for line in handle:
            rating, _, _, sent1, sent2 = line.strip().split("\t")
            if rating == "0":
                continue
            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    # return src_lang, dst_lang, pairs
    # MS makes the vocab for paraphrase the same

    return pairs, lang

def load_sentiment(split):
    """Load the Sentiment Kaggle Comp Dataset"""
    logger = logging.getLogger(f"{__name__}.load_sentiment")
    lang = Lang("sentiment")

    if split == "train":
        path = config.sentiment_train_path
    elif split == "val":
        path = config.sentiment_dev_path
    elif split == "test":
        path = config.sentiment_test_path
   
    logger.info("loading %s from %s" % (split, path))

    pairs = []
    
    with open(path) as handle:

        # skip header
        handle.readline()

        for line in handle:
            _, _, sent1, sent2 = line.strip().split("\t")

            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    return pairs, lang


def load_tamil(split):
    """Load the En to Tamil dataset, current SOTA ~13 bleu"""
    logger = logging.getLogger(f"{__name__}.load_tamil")
    lang = Lang("tamil")

    if split == "train":
        path = config.tamil_train_path
    elif split == "val":
        path = config.tamil_dev_path
    elif split == "test":
        path = config.tamil_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    with open(path) as handle:

        handle.readline()

        for line in handle:
            sent1, sent2 = line.strip().split("\t")
            #if rating == "0":
            #    continue
            sent1 = tokenize(sent1)
            #I dunno how to tokenize tamil.....?
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            pairs.append([sent1, sent2])

    return pairs, lang

def load_compression(split):
    """Load the Google Sentence Compression Dataset"""
    logger = logging.getLogger(f"{__name__}.load_compression")
    lang = Lang("compression")

    if split == "train":
        path = config.compression_train_path
    elif split == "val":
        path = config.compression_dev_path 
    elif split == "test":
        path = config.compression_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    with open(path) as handle:

        handle.readline()

        for line in handle:
            sent1, sent2 = line.strip().split("\t")
            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
           # print(len(sent1), sent1)
           # print(len(sent2), sent2)
           # print()
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            pairs.append([sent1, sent2])

    return pairs, lang

def load_stanford(split):
    """Load the Stanford Sentiment Dataset phrases"""
    logger = logging.getLogger(f"{__name__}.load_stanford")
    lang = Lang("stanford")

    if split == "train":
        path = config.stanford_train_path
    elif split == "val":
        path = config.stanford_dev_path
    elif split == "test":
        path = config.stanford_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []
    
    with open(path) as handle:

        # skip header
        #handle.readline()

        for line in handle:
            _, _, sent1, sent2 = line.strip().split("\t")

            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    return pairs, lang

def load_stanford_sent(split):
    """Load the Stanford Sentiment Dataset sentences"""
    logger = logging.getLogger(f"{__name__}.load_stanford_sent")
    lang = Lang("stanford_sent")

    if split == "train":
        path = config.stanford_sent_train_path
    elif split == "val":
        path = config.stanford_sent_dev_path
    elif split == "test":
        path = config.stanford_sent_test_path

    logger.info("loading %s from %s" % (split, path))

    pairs = []

    with open(path) as handle:

        # skip header
        #handle.readline()

        for line in handle:
            _, _, sent1, sent2 = line.strip().split("\t")

            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            lang.add_sentence(sent1)
            lang.add_sentence(sent2)

            # pairs.append([sent1, sent2, rating])
            pairs.append([sent1, sent2])

    return pairs, lang
