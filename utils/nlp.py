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
    #TODO: ma partire dai token dello step prima?????
    for tweet in tweets:
        words = TOKENIZER(tweet['text'].lower())
        #text = ''.join([c for c in text if c not in punctuation])
        #words = text.split()
#        words = [word for word in words if word not in STOPWORDS]

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

#TODO: qui da capire se passare i token precedenti o costruirne uno più grande per accogliere anche future parole mai viste (pensa a riempimento a runtime)
def gen_sequence(TOKENIZER,tweets,STOPWORDS,vocab):
    X, y = [], []
    for tweet in tweets:
        words = TOKENIZER(tweet['text'].lower())
        #text = ''.join([c for c in text if c not in punctuation])
        #words = text.split()
        #words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(int(tweet['label']))
    return X, y


def get_embedding_weights(EMBEDDING_DIM,word2vec_model, vocab):
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    #FIXME: Non leggem mai una fava da word2vec
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model.wv[k]
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
    text = re_sub(r"@\w+", "NHZ_USER")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<NHZ_SMILE>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<NHZ_LOL_FACE>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<NHZ_SAD_FACE>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<NHZ_NEUTRAL_FACE>")
    text = re_sub(r"<3","<NHZ_HEART>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NHZ_NUMBERS>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", "")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <NHZ_ELONG>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def glove_tokenize(text ,FLAGS=FLAGS):
    #TODO: implement tokenizer
    text = tokenize_nhz(text,FLAGS)
    #text = text.translate(punctuation)
    text_stripped = ""
    for c in text:
        if c not in punctuation:
            text_stripped += c
        else:
            text_stripped += " "
    text = ''.join([c for c in text if c not in punctuation])
    words = text_stripped.split()
    words = [word for word in words if word not in getStopWords()]
    return words

def getStopWords(fileName='/home/thenhz/workspace/hate-speech/data/stopwords/IT-stopwordsISO-stopwords.txt'):
    with open(fileName) as myFile:
        lines = myFile.read().splitlines()
    return lines