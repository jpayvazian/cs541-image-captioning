import numpy as np
import re
import csv
import json
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
    token_freq[vocab["<unk>"]] = 0

    return token_freq

def labels_to_json(inpath, outpath):
    '''
    Formats labels file to JSON for compatibility with metric evaluation functions
    Differs from output function since 5 label captions per image
    :param inpath: path of original labels
    :param outpath: path to save output file to
    :return: JSON file
    '''
    data = {}
    with open(inpath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        i = 0
        caption_array = []
        # convert each row into a dictionary
        for rows in csvReader:
            if i % 5 == 0:
                caption_array = []
            key = rows['image']
            caption_array.append(rows['caption'])
            data[key] = caption_array
            i += 1

    with open(outpath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

def output_to_json(inpath, outpath):
    '''
    Formats output file to JSON for compatibility with metric evaluation functions
    Differs from label function since only 1 output caption per image
    :param inpath: path of original outputs
    :param outpath: path to save output file to
    :return: JSON file
    '''
    data = {}
    with open(inpath) as fh:
        for line in fh:
            image, caption = line.strip().split(",")
            data[image] = [caption]

    del data['image']

    with open(outpath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))