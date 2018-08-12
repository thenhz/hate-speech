from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim, sklearn
from collections import defaultdict
from utils.nlp import glove_tokenize,gen_vocab,get_embedding_weights, gen_sequence
from input.twitter import select_tweets
from models import lstm as mymodel
from gensim.parsing.preprocessing import STOPWORDS

from nltk import tokenize as tokenize_nltk



### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
#freq = defaultdict(int)
tweets = {}
data_root = '/home/thenhz/workspace/hate-speech/data/'

EMBEDDING_DIM = 25
GLOVE_MODEL_FILE = data_root + 'word2vec/glove_WIKI'
SEED = 324
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = "categorical_crossentropy"
OPTIMIZER = "adagrad"
KERNEL = None
TOKENIZER = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = 'glove'#glove
LEARN_EMBEDDINGS = False
EPOCHS = 1
BATCH_SIZE = 32
SCALE_LOSS_FUN = None

word2vec_model = None
tokenizer = 'glove'#glove
inputFile = files= data_root + 'hate-speech/haspeede_TW-train.tsv'


#python lstm.py -f ~/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.25d.txt
#-d 25 --tokenizer glove --loss categorical_crossentropy --optimizer adam --initialize-weights random
#--learn-embeddings --epochs 10 --batch-size 512

np.random.seed(SEED)
print('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
print('Embedding Dimension: %d' %(EMBEDDING_DIM))
print('Allowing embedding learning: %s' %(str(LEARN_EMBEDDINGS)))

#Word2Vec.load('glove_WIKI')
#word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)
word2vec_model = gensim.models.Word2Vec.load(GLOVE_MODEL_FILE)

if tokenizer == "glove":
    TOKENIZER = glove_tokenize
elif tokenizer == "nltk":
    TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize


tweets = select_tweets(files=[inputFile],TOKENIZER=TOKENIZER,word2vec_model=word2vec_model)
vocab,reverse_vocab = gen_vocab(TOKENIZER=TOKENIZER, tweets=tweets,STOPWORDS=STOPWORDS)
#filter_vocab(20000)
X, y = gen_sequence(TOKENIZER=TOKENIZER,tweets=tweets,STOPWORDS=STOPWORDS,vocab=vocab)
#Y = y.reshape((len(y), 1))
MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
print("max seq length is %d"%(MAX_SEQUENCE_LENGTH))

data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(y)
a = np.int8(np.logical_not(y))
y = np.vstack((y,a)).T
data, y = sklearn.utils.shuffle(data, y)
W = get_embedding_weights(EMBEDDING_DIM,word2vec_model,vocab)

model = mymodel.lstm_model(data.shape[1], embedding_dim=EMBEDDING_DIM,vocab=vocab,learn_embeddings=LEARN_EMBEDDINGS,LOSS_FUN=LOSS_FUN,OPTIMIZER=OPTIMIZER)
#model = lstm_model(data.shape[1], 25, get_embedding_weights())
mymodel.train_LSTM(data, y, model, EMBEDDING_DIM, W, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   INITIALIZE_WEIGHTS_WITH=INITIALIZE_WEIGHTS_WITH,SCALE_LOSS_FUN=SCALE_LOSS_FUN,NO_OF_FOLDS=NO_OF_FOLDS)