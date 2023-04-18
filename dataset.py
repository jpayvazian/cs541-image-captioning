import numpy as np
import tensorflow as tf

class FlickrDataset(tf.keras.utils.Sequence):
    '''
    Dataset class to use for training, generates batches on-the-fly
    Shuffles data every epoch to avoid memorizing order/overfitting
    '''
    def __init__(self, df, tokenizer, vocab_size, max_len, batch_size, features, decoder_type):
        self.df = df.copy().sample(frac=1).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.features = features
        self.decoder_type = decoder_type

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        '''
        Gets batch of data, encodes captions and pads to max_len
        Y encoded as int rather than 1-hot, to use with Sparse_Categorical_Crossentropy
        Batch data format varies by decoder_type

        Transformer:
        Y seq is same as X, shifted over 1
        Example seq: 1 2 3, max_len = 4
        X_seq = 1 2 0 0 -> Y_seq = 2 3 0 0

        lstm_baseline:
        Y seq is single next word of X seq, multiple X seq per image
        Example seq: 1 2 3, max_len = 4
        X_seq = 1 0 0 0 -> Y_seq = 2
        X_seq = 1 2 0 0 -> Y_seq = 3
        X_seq = 1 2 3 0 -> Y_seq = 4

        lstm_attention:
        No X seq, only X features

        :param index: the batch number to get
        :return: (X_feature, X_seq), Y_seq for transformer/lstm_baseline
        :return: X_feature, Y_seq for lstm_attention
        '''
        X_feature, X_seq, Y_seq = [], [], []
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]

        for image, caption in zip(batch['image'], batch['caption']):
            feature = self.features[image][0]
            seq = self.tokenizer.texts_to_sequences([caption])[0]

            if self.decoder_type == 'transformer':
                x_seq, y_seq = seq[:-1], seq[1:]
                x_seq = tf.keras.utils.pad_sequences([x_seq], maxlen=self.max_len, padding='post')[0]
                y_seq = tf.keras.utils.pad_sequences([y_seq], maxlen=self.max_len, padding='post')[0]
                X_feature.append(feature)
                X_seq.append(x_seq)
                Y_seq.append(y_seq)

            elif self.decoder_type == 'lstm_baseline':
                for i in range(1, len(seq)):
                    x_seq, y_seq = seq[:i], seq[i]
                    x_seq = tf.keras.utils.pad_sequences([x_seq], maxlen=self.max_len, padding='post')[0]
                    X_feature.append(feature)
                    X_seq.append(x_seq)
                    Y_seq.append(y_seq)

            elif self.decoder_type == 'lstm_attention':
                y_seq = tf.keras.utils.pad_sequences([seq], maxlen=self.max_len, padding='post')[0]
                X_feature.append(feature)
                Y_seq.append(y_seq)

        if self.decoder_type == 'lstm_attention':
            return np.array(X_feature), np.array(Y_seq)
        else:
            return (np.array(X_feature), np.array(X_seq)), np.array(Y_seq)