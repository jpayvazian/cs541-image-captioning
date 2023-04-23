import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.models import Sequential, Model
from keras import initializers, regularizers, constraints

class Decoder_Baseline(Model):
    def __init__(self, units, max_len, embed_dim, vocab_size, dropout):
        super(Decoder_Baseline, self).__init__()
        self.units = units
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size 
        self.dropout = dropout
        self.hidden = tf.zeros((32, self.units)) # TODO we can do better than this

        self.dense1 = tf.keras.layers.Dense(self.embed_dim, activation='relu')
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim, mask_zero=False)
        self.lstm = tf.keras.layers.LSTM(self.embed_dim)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dense2 = tf.keras.layers.Dense(self.units, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)
        self.dense3 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs):
        img_features, seq_input = inputs

        img_features = self.dense1(img_features)
        img_features_reshaped = tf.keras.layers.Reshape((1, self.embed_dim), input_shape=(self.embed_dim,))(img_features)

        seq_features = self.embedding(seq_input)
        merged = tf.keras.layers.concatenate([img_features_reshaped, seq_features], axis=1)



        seq_features = self.lstm(merged)

        x = self.dropout1(seq_features)
        x = tf.keras.layers.add([x, img_features]) # TODO issue here w/ different batch sizes :/
        x = self.dense2(x)
        x = self.dropout2(x)
        output = self.dense3(x) # Removed softmax since its done in caption generation/loss fcn
        return output