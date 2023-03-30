import numpy as np
import tensorflow as tf

# Dataset class to use for training
class FlickrDataset(tf.keras.utils.Sequence):
    def __init__(self, df, tokenizer, vocab_size, max_length, batch_size, features):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.features = features

    # Shuffles data every epoch to avoid memorizing order/overfitting
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
                Create multiple training features for each "next word" in sequence
                  e.g for sentence: a b c d with max_length = 5
                  X_seq[0],Y[0] = a 0 0 0 0, b
                  X_seq[1],Y[1] = a b 0 0 0, c
                  X_seq[2],Y[2] = a b c 0 0, d
                '''
                for i in range(1, len(seq)):
                    x_seq, y_seq = seq[:i], seq[i]
                    x_seq = tf.keras.preprocessing.sequence.pad_sequences([x_seq], maxlen=self.max_length)[0]
                    y_seq = tf.keras.utils.to_categorical([y_seq], num_classes=self.vocab_size)[0]
                    X_feature.append(feature)
                    X_seq.append(x_seq)
                    Y_seq.append(y_seq)

        return np.array(X_feature), np.array(X_seq), np.array(Y_seq)