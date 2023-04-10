import tensorflow as tf
import os
import pickle
import numpy as np
from vit_keras import vit, utils
import pandas as pd

'''
Feature extraction (encoder) for images using pretrained CNN Resnet or VIT
Save/load features to .pkl to avoid rerunning this operation
:param image_files: list of string image file names

:return: dictionary of features for each image
'''
def extract_features_resnet(image_files):
    if not os.path.isfile("flickr8k/features.pkl"):
        features = {}
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    
        for file in image_files:
            image = tf.keras.utils.load_img(os.path.join('flickr8k/Images', file), target_size=(224, 224))
            image = tf.keras.utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = tf.keras.applications.resnet50.preprocess_input(image)
            feature = model.predict(image)
            features[file] = feature

        # Cache features to avoid having to run multiple times
        with open("flickr8k/features.pkl", "wb") as f:
            pickle.dump(features, f)

    # If features already cached, load them from disk
    else:
        with open("flickr8k/features.pkl", "rb") as f:
            features = pickle.load(f)

    return features

# TODO: ViT encoder
def extract_features_vit(image_files):
    if not os.path.isfile("flickr8k/features.pkl"):
        features = {}
        image_size = 224
        model = vit.vit_b16(
            image_size = image_size,
            pretrained = True,
            activation ='sigmoid',
            include_top=False,
            pretrained_top=False)

        for file in image_files:
            image = tf.keras.utils.load_img(os.path.join('flickr8k/Images', file), target_size=(224, 224))
            image = tf.keras.utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = vit.preprocess_inputs(image)
            # image = utils.read(os.path.join('flickr8k/Images', file), image_size)
            # processed_image = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
            feature = model.predict(image)
            features[file] = feature
            
        # Cache features to avoid having to run multiple times
        with open("flickr8k/features.pkl", "wb") as f:
            pickle.dump(features, f)

    # If features already cached, load them from disk
    else:
        with open("flickr8k/features.pkl", "rb") as f:
            features = pickle.load(f)

    return features
