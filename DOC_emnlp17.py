
import json
import numpy as np
np.random.seed(1)
from keras import preprocessing
fn = '50EleReviews.json'
with open(fn, 'r') as infile:
        docs = json.load(infile)
X = docs['X']
y = np.asarray(docs['y'])
num_classes = len(docs['target_names'])
def count_word(X):
    word_count = dict()
    for d in X:
        for w in d.split(' '):
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1
    return word_count
word_count = count_word(X)
print('total words: ', len(word_count))
freq_words = [w  for w, c in word_count.items() if c > 10]
print('frequent word size = ', len(freq_words))
word_to_idx = {w: i+2  for i, w in enumerate(freq_words)}
def index_word(X):
    seqs = []
    for d in X:
        seq = []
        for w in d.lower().split():
            if w in word_to_idx:
                seq.append(word_to_idx[w])
            else:
                seq.append(1) #rare word index = 1
        seqs.append(seq)
    return seqs
indexed_X = index_word(X)
padded_X = preprocessing.sequence.pad_sequences(indexed_X, maxlen=3000, dtype='int32', padding='post', truncating='post', value = 0.)
def splitTrainTest(X, y):
    shuffle_idx = np.random.permutation(len(y))
    split_idx = int(2*len(y)/3)
    shuffled_X = X[shuffle_idx]
    shuffled_y = y[shuffle_idx]
    return shuffled_X[:split_idx], shuffled_y[:split_idx], shuffled_X[split_idx:], shuffled_y[split_idx:]
train_X, train_y, test_X, test_y = splitTrainTest(padded_X, y)
from keras.utils.np_utils import to_categorical
cate_train_y = to_categorical(train_y,50)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding, Input, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
def Network(MAX_SEQUENCE_LENGTH = 3000, EMBEDDING_DIM = 300, nb_word = len(word_to_idx)+2, filter_lengths = [3, 4, 5],
    nb_filter = 150, hidden_dims =250):
    graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,  EMBEDDING_DIM))
    convs = []
    for fsz in filter_lengths:
        conv = Conv1D(filters=nb_filter,
                                 kernel_size=fsz,
                                 padding='valid',
                                 activation='relu')(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    if len(filter_lengths)>1:
        out = Concatenate(axis=-1)(convs)
    else:
        out = convs[0]
    graph = Model(inputs=graph_in, outputs=out)
    emb_layer = [Embedding(nb_word,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True),
                 Dropout(0.2)
        ]
    conv_layer = [
            graph,
        ]
    feature_layers1 = [
            Dense(hidden_dims),
            Dropout(0.2),
            Activation('relu')
    ]
    feature_layers2 = [
            Dense(50),
            Dropout(0.2),
    ]
    output_layer = [
            Activation('sigmoid')
    ]
    model = Sequential(emb_layer+conv_layer+feature_layers1+feature_layers2+output_layer)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
model = Network()
bestmodel_path = 'bestmodel.h5'
checkpointer = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
early_stopping=EarlyStopping(monitor='val_loss', patience=5)
model.fit(train_X, cate_train_y,epochs=100, batch_size=128, callbacks=[checkpointer, early_stopping], validation_split=0.2)
model.load_weights(bestmodel_path)
test_X_pred = model.predict(test_X)
test_y_pred=[]
for p in test_X_pred:# loop every test prediction
    max_class = np.argmax(p)# predicted class
    max_value = np.max(p)# predicted probability
    test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
from sklearn.metrics import accuracy_score,f1_score
acc=accuracy_score(test_y,test_y_pred)
f1=f1_score(test_y,test_y_pred,average="macro")
print(acc,f1)




