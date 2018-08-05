import csv
from sklearn.base import TransformerMixin
import re
from string import punctuation



class TW_Evalita18_Reader(TransformerMixin):

    def __init__(self, files):
        self.files = files

    def transform(self, X, *_):
        return self.get_data()

    def get_data(self):
        tweets = []
        for file in self.files:
            with open(file) as tsvFile:
                for row in csv.reader(tsvFile, dialect="excel-tab"):
                    tweets.append({
                        'id': row[0],
                        'text': row[1].lower(),
                        'label': int(row[2])
                    })
        return tweets

    def fit(self, *_):
        return self


class TW_FilterValid(TransformerMixin):

    def __init__(self, word2vecModel, tweets, tokenizer='glove'):
        self.tweets = tweets
        self.tokenizer = tokenizer
        self.word2vecModel = word2vecModel

    def transform(self, X, *_):
        X, Y = [], []
        tweet_return = []
        for tweet in self.tweets:
            _emb = 0
            words = self.tokenizer(tweet['text'].lower())
            for w in words:
                if w in self.word2vecModel:  # Check if embeeding there in GLove model
                    _emb += 1
            if _emb:  # Not a blank tweet
                tweet_return.append(tweet)
        print('Tweets total:', len(self.tweets))
        print('Tweets selected:', len(tweet_return))
        # pdb.set_trace()
        return tweet_return

    def fit(self, *_):
        return self


class CleanTweet(TransformerMixin):

    def __init__(self, word2vecModel, tweets, tokenizer='glove',flags = re.MULTILINE | re.DOTALL):
        self.eyes = r"[8:=;]"
        self.nose = r"['`\-]?"
        self.flags = flags

    def transform(self, text, *_):

        def hashtag(text):
            text = text.group()
            hashtag_body = text[1:]
            if hashtag_body.isupper():
                result = u"<hashtag> {} <allcaps>".format(hashtag_body)
            else:
                result = " ".join(["<hashtag>"])  # + re.split(u"(?=[A-Z])", hashtag_body, flags=FLAGS))
            return result

        def allcaps(text):
            text = text.group()
            return text.lower() + " <allcaps>"

        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=self.flags)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"/", " / ")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(self.eyes, self.nose, self.nose, self.eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(self.eyes, self.nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(self.eyes, self.nose, self.nose, self.eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(self.eyes, self.nose), "<neutralface>")
        text = re_sub(r"<3", "<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

        ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
        text = re_sub(r"([A-Z]){2,}", allcaps)

        text = text.lower()

        text = ''.join([c for c in text if c not in punctuation])
        return text.split()


    def fit(self, *_):
        return self