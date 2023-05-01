# cs541-image-captioning
Final Project for CS 541: Deep Learning  
Authors: Jack Ayvazian, Luke Deratzou, Jasmine Laber  
[Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

File descriptions:
* `main.py`: Main script for training pipeline 
* `encoder.py`: Encoder feature extraction 
* `decoder_baseline.py`, `decoder_lstm_attention.py`, `decoder_transformer.py`: Decoder models 
* `caption.py`: Beam search caption generation 
* `dataset.py`: Data batch generator 
* `eval.py`: Evaluation script with metrics and loss functions
* `utils.py`: Miscellaneous functions for data cleaning and frequency, feature squashing, json formatting

Included only in [Github repo](https://github.com/jpayvazian/cs541-image-captioning):
* `flickr8k/`: Labels and Output caption files from evaluation
* `turing/`: Shell script for training via WPI Turing cluster and example slurm.out training proof  

Command line args:  
ENCODER_TYPES: `resnet`, `vit`  
DECODER_TYPES: `lstm_baseline`, `lstm_attention`, `transformer`

To train: `python main.py [ENCODER_TYPE] [DECODER_TYPE]`  
To evaluate: `python eval.py [ENCODER_TYPE] [DECODER_TYPE]`
