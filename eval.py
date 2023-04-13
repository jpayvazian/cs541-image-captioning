import tensorflow as tf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json
import csv

# TODO:
#  - BLEU1-4, ROUGE-L, METEOR, CIDEr, SPICE
#  - Baseline evaluation
#  - Attention visualization]





@tf.function
def masked_loss(y, yhat):
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

def bleu():
        scorer = Bleu(n=4)
        score, scores = scorer.compute_score(labels, output)

        print('bleu = %s' % score)

def cider():
    scorer = Cider()
    score, scores = scorer.compute_score(labels, output)
    
    print('cider = %s' % score)

def meteor():
    scorer = Meteor()
    score, scores = scorer.compute_score(labels, output)

    print('cider = %s' % score)

def rouge():
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)

def spice():
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)

def make_labels_json(inpath, outpath):

        #dictionary
        data = {}

        #use csv reader
        with open(inpath,encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)

            i = 0
            caption_array = []
            #convert each row into a dictionary
            for rows in csvReader:
                # i +=1
                # print(i)
                # if i % 5 != 0:
                #       caption_array = []
                key = rows['image']
                data[key] = caption_array.append(rows['caption'])
                print(data[key])
        
         # function to dump data
        with open(outpath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data, indent=4))

if __name__ == "__main__":

     #make function so that you can load all seperately, use 75_1 as example

    #load in the output and labels
    csv_file = r'flickr8k/Labels/captions_test75.csv'
    json_file = r'flickr8k/Output/captions_transformer75_1.json'

    #.txt to json conversion
    #one for labels

    make_labels_json(csv_file, json_file)
   


    #one for outputs

    #lets pretend we do that lol



    #have to duplicate the produced labels by 5
    #Doesn't look like I have to do this lol
    #{"1909090": ["train traveling down a track,,,"]}






