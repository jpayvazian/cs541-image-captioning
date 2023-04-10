import numpy as np
import re
from collections import Counter

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


def get_freq(captions, vocab):
    '''
    Gets the frequency of each token in the caption labels
    Sets frequency of <start> token to 0, (pad tokens also excluded since not in original caption list)
    :return: np.array size=vocab_size where values are token freq, index corresponding to token ID
    '''
    counts = Counter(" ".join(captions).split())
    token_freq = np.zeros(len(vocab) + 1)

    for token, count in counts.items():
        index = vocab[token]
        token_freq[index] = count

    token_freq[vocab["<start>"]] = 0

    return token_freq