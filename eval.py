import tensorflow as tf
import evaluate
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import json
import csv
import sys

@tf.function
def masked_loss_transformer(y, yhat):
    '''
    Custom cross entropy loss which uses mask to exclude pad/<start> tokens in calculation
    Sparse Categorical since labels are integer encodings (not 1-hot)
    '''
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y, yhat)
    mask = (y != 0) & (loss < 1e8)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

@tf.function
def masked_loss_lstm(y, yhat):
    '''
    Same as other masked loss, except reduce_mean used to prevent divide by 0
    '''
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y, yhat)
    mask = (y != 0) & (loss < 1e8)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

@tf.function
def masked_acc(y, yhat):
    '''
    Custom accuracy which uses mask to exclude pad/<start> tokens in calculation
    Checks equality between most probable next token and label
    '''
    mask = tf.cast(y != 0, tf.float32)
    yhat = tf.argmax(yhat, axis=-1)
    y = tf.cast(y, yhat.dtype)
    correct = tf.cast(yhat == y, mask.dtype)
    return tf.reduce_sum(correct * mask)/tf.reduce_sum(mask)

def bleu(labels, output):
    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(labels, output)
    print('bleu = %s' % score)

def rouge(labels, output):
    scorer = Rouge()
    score, scores = scorer.compute_score(labels, output)
    print('rouge = %s' % score)

def cider(labels, output):
    scorer = Cider()
    score, scores = scorer.compute_score(labels, output)
    print('cider = %s' % score)

def meteor(labels, output):
    scorer = evaluate.load('meteor')
    labels = list(labels.values())
    output = [caption for img_file in output for caption in output[img_file]]
    score = scorer.compute(predictions=output, references=labels)
    print(score)

def make_labels_json(inpath, outpath):
    # dictionary
    data = {}

    # use csv reader
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

    # function to dump data
    with open(outpath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

def make_output_json(inpath, outpath):
    # create dictionary
    dict = {}

    with open(inpath) as fh:
        for line in fh:
            image, caption = line.strip().split(",")
            dict[image] = [caption]

    del dict['image']

    with open(outpath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(dict, indent=4))

ENCODER_TYPES = ['resnet', 'vit']
DECODER_TYPES = ['transformer', 'lstm_baseline', 'lstm_attention']
if __name__ == "__main__":
    '''
    COMMAND LINE ARGS:
    python eval.py [ENCODER_TYPE] [DECODER_TYPE]
    '''
    ENCODER_TYPE, DECODER_TYPE = sys.argv[1], sys.argv[2]
    if (ENCODER_TYPE not in ENCODER_TYPES) or (DECODER_TYPE not in DECODER_TYPES):
        print("Invalid encoder/decoder type")
        sys.exit(1)

    # load in the output and labels
    label_raw = r'flickr8k/Labels/captions_test.csv'
    label_json = r'flickr8k/Labels/captions_test.json'

    output_raw = f'flickr8k/Output/captions_{ENCODER_TYPE}_{DECODER_TYPE}.txt'
    output_json = f'flickr8k/Output/captions_{ENCODER_TYPE}_{DECODER_TYPE}.json'

    # labels to json conversion
    make_labels_json(label_raw, label_json)

    # output to json conversion
    make_output_json(output_raw, output_json)

    # evaluation on metrics
    with open(label_json, 'r') as file:
        caption_labels = json.load(file)

    with open(output_json, 'r') as file:
        caption_output = json.load(file)

    bleu(caption_labels, caption_output)
    rouge(caption_labels, caption_output)
    cider(caption_labels, caption_output)
    meteor(caption_labels, caption_output)