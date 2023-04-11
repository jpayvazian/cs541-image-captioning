import tensorflow as tf
import os
import pickle
import numpy as np
from vit_keras import vit
from utils import flatten_features

def extract_features(image_files, encoder_type):
    '''
    Feature extraction (encoder) for images using pretrained CNN Resnet or ViT
    Save/load features to .pkl to avoid rerunning this operation
    Flatten w.h feature dims for compatibility in decoder
    :param image_files: list of string image file names
    :param encoder_type: string for which model to use, 'resnet' or 'vit'

    :return: dictionary of features for each image
    '''
    features_path = os.path.join('flickr8k', 'Features', f'features_{encoder_type}.pkl')
    if not os.path.isfile(features_path):
        model, preprocessor, flattener = None, None, None
        features = {}

        if encoder_type == 'resnet':
            model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
            preprocessor = tf.keras.applications.resnet50.preprocess_input
            flattener = flatten_features #(1,7,7,2048) -> (1,49,2048)
        elif encoder_type == 'vit':
            vit_base = vit.vit_b16(pretrained=True, include_top=False, pretrained_top=False)
            model = tf.keras.Model(inputs=vit_base.input, outputs=vit_base.layers[-2].output)
            preprocessor = vit.preprocess_inputs
            flattener = lambda x: x #(1,197,768)

        for file in image_files:
            image = tf.keras.utils.load_img(os.path.join('flickr8k/Images', file), target_size=(224, 224))
            image = tf.keras.utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocessor(image)
            feature = model.predict(image)
            features[file] = feature

        features = flattener(features)

        # Cache features to avoid having to run multiple times
        with open(features_path, "wb") as f:
            pickle.dump(features, f)

    # If features already cached, load them from disk
    else:
        with open(features_path, "rb") as f:
            features = pickle.load(f)

    return features
