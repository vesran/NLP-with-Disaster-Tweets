from src.preprocessing import clean
import tensorflow as tf
import pandas as pd
import numpy as np
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
filename = './data/entire_corpus.csv'
df = pd.read_csv(filename)
df_train = df[df['source'] == 'train'].copy()
df_test = df[df['source'] == 'test'].copy()

# Clean
df_train['text'] = df_train['text'].apply(clean)
df_test['text'] = df_test['text'].apply(clean)

# Encoding
vocab, max_length = get_vocabulary(df_train['text'])

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_train['text'])
sequences_train = tokenizer.texts_to_sequences(df_train['text'])
sequences_test = tokenizer.texts_to_sequences(df_test['text'])

# Padding
padded_seqs_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length,
                                                            padding='post', truncating='post')
padded_seqs_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length,
                                                            padding='post', truncating='post')

# Data sets
X_train = padded_seqs_train
X_test = padded_seqs_test
y_train = df_train['target'].values
y_test = df_test['target'].values

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(100)
class_weights = {0: 1-((y_train == 0).astype(int).sum()/y_train.shape[0]),
                 1: 1-((y_train == 1).astype(int).sum()/y_train.shape[0])}


# Model
model = tf.keras.Sequential()
model.add(a := tf.keras.layers.Embedding(input_dim=len(vocab)+1, output_dim=100, input_length=max_length))
model.add(b := tf.keras.layers.Conv1D(filters=200, kernel_size=5))
model.add(c := tf.keras.layers.MaxPool1D())
model.add(d := tf.keras.layers.Conv1D(filters=200, kernel_size=4))
model.add(e := tf.keras.layers.MaxPool1D())
model.add(f := tf.keras.layers.Conv1D(filters=200, kernel_size=3))
model.add(g := tf.keras.layers.MaxPool1D())
model.add(h := tf.keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=test_ds, class_weight=class_weights)

# Score : ~0.74
