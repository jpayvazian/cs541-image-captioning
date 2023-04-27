# cs541-image-captioning
Final Project for CS 541: Deep Learning

[Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Command line args:

ENCODER_TYPES:
* 'resnet'
* 'vit'

DECODER_TYPES:
* 'transformer'
* 'lstm_baseline'
* 'lstm_attention'

To train: `python main.py [ENCODER_TYPE] [DECODER_TYPE]`

To evaluate: `python eval.py [ENCODER_TYPE] [DECODER_TYPE]`