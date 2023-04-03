import numpy as np
import tensorflow as tf

class FlickrDataset(tf.keras.utils.Sequence):
    '''
    Dataset class to use for training, generates batches on-the-fly
    Shuffles data every epoch to avoid memorizing order/overfitting
    Encodes and pads caption label sequences
    X = (image features, seq), Y = next word in seq
    Y encoded as int rather than 1-hot, to use with Sparse_Categorical_Crossentropy
    '''
    def __init__(self, df, tokenizer, vocab_size, max_len, batch_size, features):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.features = features

    # TODO: Generate caption for 1 particular image on epoch end to see progress
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df) // self.batch_size

    # Get batch of data
    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]
        X_feature, X_seq, Y_seq = self.__get_batch(batch)
        return (X_feature, X_seq), Y_seq

    def __get_batch(self, batch):
        X_feature, X_seq, Y_seq = [], [], []
        image_files = batch['image'].tolist()

        # For each image in batch, get extracted features and all associated captions
        for image in image_files:
            feature = self.features[image][0]
            captions = batch.loc[batch['image'] == image, 'caption'].tolist()

            # Encode each caption
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                '''
                Create multiple training features for each "next word" in sequence, y label is same as x shifted once
                  e.g for sentence: a b c d with max_len = 5
                  X_seq[0],Y[0] = a 0 0 0 0, b 0 0 0 0
                  X_seq[1],Y[1] = a b 0 0 0, b c 0 0 0
                  X_seq[2],Y[2] = a b c 0 0, b c d 0 0
                '''
                for i in range(1, len(seq)):
                    x_seq, y_seq = seq[:i], seq[1:i+1]
                    x_seq = tf.keras.utils.pad_sequences([x_seq], maxlen=self.max_len)[0]
                    y_seq = tf.keras.utils.pad_sequences([y_seq], maxlen=self.max_len)[0]
                    X_feature.append(feature)
                    X_seq.append(x_seq)
                    Y_seq.append(y_seq)

        return np.array(X_feature), np.array(X_seq), np.array(Y_seq)