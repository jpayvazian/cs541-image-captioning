import numpy as np
import tensorflow as tf

class FlickrDataset(tf.keras.utils.Sequence):
    '''
    Dataset class to use for training, generates batches on-the-fly
    Shuffles data every epoch to avoid memorizing order/overfitting
    '''
    def __init__(self, df, tokenizer, vocab_size, max_len, batch_size, features):
        self.df = df.copy().sample(frac=1).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.features = features

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        '''
        Gets batch of data, encodes captions and pads to max_len
        Y encoded as int rather than 1-hot, to use with Sparse_Categorical_Crossentropy
        Y seq is same as X, shifted over 1

        Example seq: 1 2 3 4 5, max_len = 6
        X_seq = 1 2 3 4 0 0
        Y_seq = 2 3 4 5 0 0

        :param index: the batch number to get
        :return: (X_feature, X_seq), Y_seq
        '''
        X_feature, X_seq, Y_seq = [], [], []
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]

        for image, caption in zip(batch['image'], batch['caption']):
            feature = self.features[image][0]
            seq = self.tokenizer.texts_to_sequences([caption])[0]

            x_seq, y_seq = seq[:-1], seq[1:]
            x_seq = tf.keras.utils.pad_sequences([x_seq], maxlen=self.max_len, padding='post')[0]
            y_seq = tf.keras.utils.pad_sequences([y_seq], maxlen=self.max_len, padding='post')[0]
            X_feature.append(feature)
            X_seq.append(x_seq)
            Y_seq.append(y_seq)

        return (np.array(X_feature), np.array(X_seq)), np.array(Y_seq)