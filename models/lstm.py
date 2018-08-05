from sklearn.model_selection import KFold
from keras.layers import Embedding, Input, LSTM
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from keras.utils import np_utils
import math



def batch_gen(X, batch_size):
    n_batches = X.shape[0] / float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0] / float(batch_size)) * batch_size
    n = 0
    for i in range(0, n_batches):
        if i < n_batches - 1:
            batch = X[i * batch_size:(i + 1) * batch_size, :]
            yield batch

        else:
            batch = X[end:, :]
            n += X[end:, :].shape[0]
            yield batch

def lstm_model(sequence_length, vocab, embedding_dim,learn_embeddings,LOSS_FUN,OPTIMIZER):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, embedding_dim, input_length=sequence_length, trainable=learn_embeddings))
    model.add(Dropout(0.25))  # , input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    print(model.summary())
    return model


def train_LSTM(X, y, model, inp_dim, weights, epochs, batch_size,INITIALIZE_WEIGHTS_WITH,SCALE_LOSS_FUN,NO_OF_FOLDS):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print
    cv_object
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print
            "ERROR!"
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0] / float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0] / float(len(y_temp))

                try:
                    y_temp = np_utils.to_categorical(y_temp, nb_classes=2)
                except Exception as e:
                    print
                    e
                    print
                    y_temp
                print
                x.shape, y.shape
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                print
                loss, acc

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print
        classification_report(y_test, y_pred)
        print
        precision_recall_fscore_support(y_test, y_pred)
        print
        y_pred
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print("macro results are")
    print("average precision is %f" % (p / NO_OF_FOLDS))
    print("average recall is %f" % (r / NO_OF_FOLDS))
    print("average f1 is %f" % (f1 / NO_OF_FOLDS))

    print("micro results are")
    print("average precision is %f" % (p1 / NO_OF_FOLDS))
    print("average recall is %f" % (r1 / NO_OF_FOLDS))
    print("average f1 is %f" % (f11 / NO_OF_FOLDS))


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)