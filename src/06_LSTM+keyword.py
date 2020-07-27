from dev.preprocessing import clean
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import os


def get_vocabulary(iter_sents):
    vocab = set()
    max_length = 0
    for text in iter_sents:
        tokens = text.split(' ')
        if len(tokens) > max_length:
            max_length = len(tokens)
        vocab.update(tokens)
    return vocab, max_length


# Read & split train test
filename = './data/clean_entire_corpus.csv'
df = pd.read_csv(filename)
df_train = df[df['source'] == 'train'].copy()
df_test = df[df['source'] == 'test'].copy()

df_train['keyword'] = df_train['keyword'].fillna('_')
df_test['keyword'] = df_test['keyword'].fillna('_')
df_train['keyword'] = df_train['keyword'].apply(lambda x: re.sub('%20', ' ', x))
df_test['keyword'] = df_test['keyword'].apply(lambda x: re.sub('%20', ' ', x))

# Encoding
_, max_length = get_vocabulary(df_train['text'])
keywords = []

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_train['clean_text'])
sequences_train = tokenizer.texts_to_sequences(df_train['clean_text'])
sequences_test = tokenizer.texts_to_sequences(df_test['clean_text'])

sequences_keyword_train = tokenizer.texts_to_sequences(df_train['keyword'])
sequences_keyword_test = tokenizer.texts_to_sequences(df_test['keyword'])

# Padding
padded_seqs_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length,
                                                            padding='post', truncating='post')
padded_seqs_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length,
                                                            padding='post', truncating='post')

padded_seqs_keyword_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_keyword_train, maxlen=2,
                                                                          padding='post')
padded_seqs_keyword_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_keyword_test, maxlen=2,
                                                                          padding='post')

# Data sets
X_train = (padded_seqs_train, padded_seqs_keyword_train)
X_test = (padded_seqs_test, padded_seqs_keyword_test)
y_train = df_train['target'].values
y_test = df_test['target'].values

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(100)

# Build an index for embeddings
embeddings_index = {}
EMBEDDINGS_PATH = '/datascience/embeddings'
EMBEDDINGS_LENGTH = 50
print('Reading lines')
with open(os.path.join(EMBEDDINGS_PATH, 'glove.6B.50d.txt'), encoding='utf-8') as f:
    lines = f.readlines()

print('Extract embeddings')
for line in lines:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print('Create embeddings matrix')
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDINGS_LENGTH))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

################################################################################
# Model
################################################################################
input_tweet = tf.keras.layers.Input(shape=(max_length, ))
input_keyword = tf.keras.layers.Input(shape=(2, ))

embeddings = tf.keras.layers.Embedding(input_dim=len(word_index)+1, input_length=max_length,
                                       output_dim=EMBEDDINGS_LENGTH, weights=[embedding_matrix], trainable=False)

# Tweets
x = embeddings(input_tweet)
x = tf.keras.layers.LSTM(128)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.Model(inputs=input_tweet, outputs=x)

# Keyword
y = embeddings(input_keyword)
y = tf.keras.layers.Flatten()(y)
y = tf.keras.Model(inputs=input_keyword, outputs=y)

concat = tf.keras.layers.concatenate((x.output, y.output))

z = tf.keras.layers.Dense(64, activation='relu')(concat)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(z)
model = tf.keras.Model([input_tweet, input_keyword], outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=test_ds)


# df_sub = pd.read_csv('./data/sample_submission.csv')
# pred = np.around(model.predict(X_test).reshape(-1, )).astype(int)
# df_sub['target'] = pred
#
# df_sub.to_csv('./sub.234.csv', index=False)


# Score : ~0.80

