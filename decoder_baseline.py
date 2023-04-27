import tensorflow as tf

class Decoder_Baseline(tf.keras.Model):
    '''
    Baseline LSTM decoder adapted from https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstms
    '''
    def __init__(self, encoder, units, embed_dim, vocab_size, dropout):
        super(Decoder_Baseline, self).__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        self.lstm = tf.keras.layers.LSTM(embed_dim)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.d1 = tf.keras.layers.Dense(units, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.d2 = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        img_features, seq_input = inputs
        img_features = self.encoder(img_features)
        img_features_reshaped = tf.keras.layers.Reshape((1, self.embed_dim), input_shape=(self.embed_dim,))(img_features)

        seq_features = self.embedding(seq_input)
        merged = tf.keras.layers.concatenate([img_features_reshaped, seq_features], axis=1)
        seq_features = self.lstm(merged)

        x = self.dropout1(seq_features)
        x = tf.keras.layers.add([x, img_features])
        x = self.d1(x)
        x = self.dropout2(x)
        return self.d2(x)
