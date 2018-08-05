import json
import pdb
import codecs
import pdb
import csv
from string import punctuation
from collections import defaultdict
import numpy as np
import re

FLAGS = re.MULTILINE | re.DOTALL




def gen_vocab(TOKENIZER, tweets, STOPWORDS):
    vocab, reverse_vocab = {}, {}
    # Processing
    vocab_index = 1
    freq = defaultdict(int)

    for tweet in tweets:
        text = TOKENIZER(tweet['text'].lower())
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word  # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    return vocab, reverse_vocab


# def filter_vocab(k):
#     global freq, vocab
#     pdb.set_trace()
#     freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
#     tokens = freq_sorted[:k]
#     vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
#     vocab['UNK'] = len(vocab) + 1


def gen_sequence(TOKENIZER,tweets,STOPWORDS,vocab):
    X, y = [], []
    for tweet in tweets:
        text = TOKENIZER(tweet['text'].lower())
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(int(tweet['label']))
    return X, y


def get_embedding_weights(EMBEDDING_DIM,word2vec_model, vocab):
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print
    "%d embedding missed" % n

    return embedding