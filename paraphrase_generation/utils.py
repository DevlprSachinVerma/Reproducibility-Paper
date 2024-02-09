import json
import logging
import math
import os
import random
import re
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torch
import torch.nn as nn

import config

from transformers import BertModel, BertTokenizer
import torch
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


plt.switch_backend("agg")


def load_glove(vocabulary):
    logger = logging.getLogger(f"{__name__}.load_glove")
    logger.info("loading embeddings")
    try:
        with open(f"glove.cache") as h:
            cache = json.load(h)
    except:
        logger.info("cache doesn't exist")
        cache = {}
        cache[config.PAD] = [0] * 768
        cache[config.SOS] = [0] * 768
        cache[config.EOS] = [0] * 768
        cache[config.UNK] = [0] * 768
        cache[config.NOFIX] = [0] * 768
    else:
        logger.info("cache found")

    cache_miss = False

    if not set(vocabulary) <= set(cache):
        cache_miss = True
        logger.warn("cache miss, loading full embeddings")
        data = {}
        with open("/kaggle/input/dataset/glove.840B.300d.txt") as h:
            for line in h:
                word, *emb = line.strip().split()
                try:
                    data[word] = [float(x) for x in emb]
                except:
                    continue
        logger.info("finished loading full embeddings")
        for word in vocabulary:
            try:
                cache[word] = data[word]
            except KeyError:
                cache[word] = [0] * 768
        logger.info("cache updated")

    embeddings = []
    for word in vocabulary:
        embeddings.append(torch.tensor(cache[word], dtype=torch.float32))
    embeddings = torch.stack(embeddings)

    if cache_miss:
        with open(f"glove.cache", "w") as h:
            json.dump(cache, h)
        logger.info("cache saved")

    return embeddings

def load_bert_embeddings(vocabulary, bert_model=model, tokenizer=tokenizer_bert):
    logger = logging.getLogger(f"{__name__}.load_bert_embeddings")
    logger.info("loading embeddings")
    
    # You might need to modify the path to the cache file depending on your setup
    cache_path = "bert.cache"
    
    try:
        with open(cache_path) as h:
            cache = json.load(h)
    except FileNotFoundError:
        logger.info("cache doesn't exist")
        cache = {}
        cache['[PAD]'] = [0] * 768 # Assuming 'bert-base-uncased' model for simplicity
        cache['[UNK]'] = [0] * 768
        cache['[NOFIX]'] = [0] * 768
        cache['[CLS]'] = [0] * 768
        cache['[SEP]'] = [0] * 768
    else:
        logger.info("cache found")

    cache_miss = False

    if not set(vocabulary) <= set(cache):
        cache_miss = True
        logger.warn("cache miss, loading full embeddings")

        for word in vocabulary:
            if word not in cache:
                # Embedding for [CLS] and [SEP] tokens is set to zero for simplicity
                if word in ['[CLS]', '[SEP]']:
                    cache[word] = [0] * 768
                else:
                    # Get BERT embedding for the word
                    embeddings = get_bert_embedding(word)
                    cache[word] = embeddings.tolist()

        logger.info("finished loading full embeddings")

    embeddings = []
    for word in vocabulary:
        embeddings.append(torch.tensor(cache[word], dtype=torch.float32))
    embeddings = torch.stack(embeddings)

    if cache_miss:
        with open(cache_path, "w") as h:
            json.dump(cache, h)
        logger.info("cache saved")

    return embeddings

def get_bert_embedding(word, bert_model=model, tokenizer=tokenizer_bert):
    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Convert tokens to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_ids).unsqueeze(0)

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        last_hidden_states = outputs.last_hidden_state

    
    return last_hidden_states.squeeze(0).mean(dim=0) 


def tokenize(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = s.split(" ")
    return s


def indices_from_sentence(word2index, sentence, unknown_threshold):
    if unknown_threshold:
        return [
            word2index.get(
                word if random.random() > unknown_threshold else config.UNK,
                word2index[config.UNK],
            )
            for word in sentence
        ]
    else:
        return [
            word2index.get(word, word2index[config.UNK]) for word in sentence
        ]


def tensor_from_sentence(word2index, sentence, unknown_threshold):
    # indices = [config.SOS]
    indices = indices_from_sentence(word2index, sentence, unknown_threshold)
    indices.append(word2index[config.EOS])
    return torch.tensor(indices, dtype=torch.long, device=config.DEV)


def tensors_from_pair(word2index, pair, shuffle, unknown_threshold):

    tensors = [
        tensor_from_sentence(word2index, pair[0], unknown_threshold),
        tensor_from_sentence(word2index, pair[1], unknown_threshold),
        tensor_from_sentence(word2index, pair[2], unknown_threshold),
    ]
    if shuffle:
        random.shuffle(tensors)
    return tensors


def bleu(reference, hypothesis, n=4): #not sure if this actually changes the n gram
    if n < 1:
        return 0
    weights = [1/n]*n
    return sentence_bleu([reference], hypothesis, weights)


def pair_iter(pairs, word2index, shuffle=False, shuffle_pairs=False, unknown_threshold=0.00):
    if shuffle:
        pairs = pairs.copy()
        random.shuffle(pairs)
    for pair in pairs:
        tensor1, tensor2, tensor3= tensors_from_pair(word2index, (pair[0], pair[1], pair[2]), shuffle_pairs, unknown_threshold)
        yield (tensor1,), (tensor2,), (tensor3,)


def sent_iter(sents, word2index, unknown_threshold=0.00):
    for sent in sents:
        tensor = tensor_from_sentence(word2index, sent, unknown_threshold)
        yield (tensor,)


def batch_iter(pairs, word2index, batch_size, shuffle=False, unknown_threshold=0.00):
    for i in range(len(pairs) // batch_size):
        batch = pairs[i : i + batch_size]
        if len(batch) != batch_size:
            continue
        batch_tensors = [
            tensors_from_pair(word2index, (pair[0], pair[1]), shuffle, unknown_threshold)
            for pair in batch
        ]

        tensors1, tensors2 = zip(*batch_tensors)

        # targets = torch.tensor(targets, dtype=torch.long, device=config.DEV)

        # tensors1_lengths = [len(t) for t in tensors1]
        # tensors2_lengths = [len(t) for t in tensors2]

        # tensors1 = nn.utils.rnn.pack_sequence(tensors1, enforce_sorted=False)
        # tensors2 = nn.utils.rnn.pack_sequence(tensors2, enforce_sorted=False)

        yield tensors1, tensors2


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<__EOS__>"], rotation=90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print("input =", input_sentence)
    print("output =", " ".join(output_words))
    showAttention(input_sentence, output_words, attentions)


def save_model(model, word2index, path):
    if not path.endswith(".tar"):
        path += ".tar"
    torch.save(
        {"weights": model.state_dict(), "word2index": word2index},
        path,
    )


def load_model(path):
    checkpoint = torch.load(path)
    return checkpoint["weights"], checkpoint["word2index"]


def extend_vocabulary(word2index, langs):
    for lang in langs:
        for word in lang.word2index:
            if word not in word2index:
                word2index[word] = len(word2index)
    return word2index
