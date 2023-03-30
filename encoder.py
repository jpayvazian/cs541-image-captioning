import tensorflow as tf
import os
import pickle
import numpy as np

'''
Feature extraction (encoder) for images using pretrained CNN Resnet
Save/load features to .pkl to avoid rerunning this operation
:param image_files: list of string image file names

:return: dictionary of features for each image
'''
def extract_features(image_files):
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