import csv

def get_data(files):
    tweets = []
    for file in files:
        with open(file) as tsvFile:
            for row in csv.reader(tsvFile, dialect="excel-tab"):
                tweets.append({
                    'id': row[0],
                    'text': row[1].lower(),
                    'label': int(row[2])
                })
    return tweets


def select_tweets(TOKENIZER,word2vec_model,files=['./haspeede_TW-train.tsv']):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data(files)
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = TOKENIZER(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:  # Not a blank tweet
            tweet_return.append(tweet)
    print
    'Tweets selected:', len(tweet_return)
    # pdb.set_trace()
    return tweet_return