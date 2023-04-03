import tensorflow as tf
import numpy as np
import re

def preprocess_text(text):
    '''
    Removes puncuation/numeric/special/extra whitespace/single chars from original 'captions.txt'
    Adds <start> and <end> tags used later for tokenizer
    :param text: caption text string to process
    :return: cleaned caption

    Example usage:
        labels = pd.read_csv('flickr8k/captions.txt')
        labels['caption'] = labels['caption'].apply(preprocess_text)
        labels.to_csv('flickr8k/captions_clean.csv', index=False)
    '''
    text = re.sub(r'[^a-z ]+', '', text.lower())
    text = '<start> ' + " ".join([word for word in text.split() if len(word) > 1]) + ' <end>'

    return text


def flatten_features(features):
    '''
    Reshapes the extracted features from encoder so the w/h dims are merged
    Flattened shape necessary for cross attention layers
    '''
    return dict((k, np.reshape(v, (1, -1, v.shape[3]))) for k, v in features.items())


def masked_loss(y, yhat):
    '''
    Custom cross entropy loss which uses mask to exclude pad tokens in calculation
    Sparse Categorical since labels are integer encodings (not 1-hot)
    '''
    loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fcn(y, yhat)
    mask = tf.cast(y != 0, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)