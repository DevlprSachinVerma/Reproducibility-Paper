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
        cache[config.PAD] = [0] * 300
        cache[config.SOS] = [0] * 300
        cache[config.EOS] = [0] * 300
        cache[config.UNK] = [0] * 300
        cache[config.NOFIX] = [0] * 300
    else:
        logger.info("cache found")

    cache_miss = False

    if not set(vocabulary) <= set(cache):
        cache_miss = True
        logger.warn("cache miss, loading full embeddings")
        data = {}
        with open('/kaggle/input/glove-sent/glove.840B.300d.txt') as h:
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
                cache[word] = [0] * 300
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
    indices = indices_from_sentence(word2index, sentence, unknown_threshold)
    return torch.tensor(indices, dtype=torch.long, device=config.DEV)


def tensors_from_pair(word2index, pair, shuffle, unknown_threshold):
    tensors = [
        tensor_from_sentence(word2index, pair[0], unknown_threshold),
        tensor_from_sentence(word2index, pair[1], unknown_threshold),
    ]
    if shuffle:
        random.shuffle(tensors)
    return tensors


def bleu(reference, hypothesis, n=4):
    if n < 1:
        return 0
    weights = [1/n]*n
    return sentence_bleu([reference], hypothesis, weights)


def pair_iter(pairs, word2index, shuffle=False, shuffle_pairs=False, unknown_threshold=0.00):
    if shuffle:
        pairs = pairs.copy()
        random.shuffle(pairs)
    for pair in pairs:
        tensor1, tensor2 = tensors_from_pair(word2index, (pair[0], pair[1]), shuffle_pairs, unknown_threshold)
        yield (tensor1,), (tensor2,)


def sent_iter(sents, word2index, batch_size, unknown_threshold=0.00):
    for i in range(len(sents)//batch_size+1):
        raw_sents = [x[0] for x in sents[i*batch_size:i*batch_size+batch_size]]
        _sents = [tensor_from_sentence(word2index, sent, unknown_threshold) for sent, target in sents[i*batch_size:i*batch_size+batch_size]]
        _targets = [torch.tensor(target, dtype=torch.long).to(config.DEV) for sent, target in sents[i*batch_size:i*batch_size+batch_size]]
        if raw_sents and _sents and _targets:
            yield(raw_sents, _sents, _targets)


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
