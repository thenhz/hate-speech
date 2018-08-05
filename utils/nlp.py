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



def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = u"<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"]) #+ re.split(u"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"



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

def tokenize_nhz(text,FLAGS = FLAGS):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def glove_tokenize(text,FLAGS=FLAGS):
    #TODO: implement tokenizer
    text = tokenize_nhz(text,FLAGS)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    #words = [word for word in words if word not in STOPWORDS]
    return words