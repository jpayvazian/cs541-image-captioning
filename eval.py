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

def bleu(labels, output):
        scorer = Bleu(n=4)
        score, scores = scorer.compute_score(labels, output)

        print('bleu = %s' % score)

def cider(labels, output):
    scorer = Cider()
    score, scores = scorer.compute_score(labels, output)
    
    print('cider = %s' % score)

def meteor(labels, output):
    scorer = Meteor()
    score, scores = scorer.compute_score(labels, output)

    print('cider = %s' % score)

def rouge(labels, output):
    scorer = Rouge()
    score, scores = scorer.compute_score(labels, output)
    print('rouge = %s' % score)

def spice(labels, output):
    scorer = Spice()
    score, scores = scorer.compute_score(labels, output)
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
                if i % 5 == 0:
                      caption_array = []
                key = rows['image']
                caption_array.append(rows['caption'])
                data[key] = caption_array
                i +=1
        
         # function to dump data
        with open(outpath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data, indent=4))
    

def make_output_json(inpath, outpath):
     
    #create dictionary
    dict = {}

    with open(inpath) as fh:
          for line in fh:
               image, caption = line.split(",")
               dict[image] = caption
    
    with open(outpath, 'w', encoding= 'utf-8') as jsonf:
         jsonf.write(json.dumps(dict, indent=4))

def print_metrics(labels, output):
    bleu(labels, output)
    cider(labels, output)
    meteor(labels, output)
    rouge(labels, output)
    spice(labels, output)

if __name__ == "__main__":

     #make function so that you can load all seperately, use 75_1 as example

    #load in the output and labels
    csv_file = r'flickr8k/Labels/captions_test75.csv'
    label_json = r'flickr8k/Output/captions_transformer75_1.json'

    txt_file = r'flickr8k/Output/captions_transformer75_1.txt'
    output_json = r'flickr8k/Output/captions_test75.json'

    #labels to json conversion
    make_labels_json(csv_file, label_json)
    
    #output to json conversion
    make_output_json(txt_file, output_json)

    #evaluation on metrics
    with open('flickr8k/Output/captions_transformer75_1.json', 'r') as file:
         labels = json.load(file)
    
    with open('flickr8k/Output/captions_test75.json', 'r') as file:
         output = json.load(file)
    
    print_metrics(labels, output)


   








