import pandas as pd
import tensorflow as tf
from encoder import extract_features
from dataset import FlickrDataset
from decoder_transformer import TransformerDecoder
from caption import Captioner, CaptionCallback
from utils import get_freq
from eval import masked_loss, masked_acc
from decoder import LSTMDecoder
# from decoder import CustomDataGenerator
import numpy as np

NUM_DECODER_LAYERS = 2
EMBEDDING_DIM = 256
NUM_HEADS = 2
DROPOUT = 0.5
EPOCHS = 10
BATCH_SIZE = 32

if __name__ == "__main__":
    # Load data
    labels = pd.read_csv('flickr8k/Labels/captions_clean.csv')[:100] # smaller size to test easier
    captions = labels['caption'].tolist()
    image_files = labels['image'].unique().tolist()

    # Train/test/valid split 80/10/10
    train_idx, test_idx = int(len(image_files) * 0.8), int(len(image_files) * 0.9)
    train_files, test_files, valid_files = image_files[:train_idx], image_files[train_idx:test_idx], image_files[test_idx:]

    # Create separate df for train/test/valid labels
    train_labels = labels[labels['image'].isin(train_files)]
    train_captions = train_labels['caption'].tolist()
    test_labels = labels[labels['image'].isin(test_files)]
    valid_labels = labels[labels['image'].isin(valid_files)]
    train_labels.reset_index(drop=True)
    test_labels.reset_index(drop=True)
    valid_labels.reset_index(drop=True)

    # TODO del below
    # train_labels = train_labels[0:5]

    # Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(train_captions)
    # Number unique words in vocab (for num classes), biggest caption size (for padding)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(caption.split()) for caption in captions)

    # Feature extraction (run through resnet)
    features = extract_features(image_files)
    # (1, 49, 2048) -> (1, 2048)
    features = dict((k, tf.keras.layers.GlobalAveragePooling2D()(np.expand_dims(v, axis=1)).numpy()) for k, v in features.items())

    # Create datasets to serve as batch generator during training
    flickr_train_data = FlickrDataset(df=train_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features)
    flickr_valid_data = FlickrDataset(df=valid_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features)

    """
    # Create transformer decoder
    freq_dist = get_freq(train_captions, tokenizer.word_index)
    transformer = TransformerDecoder(freq_dist=freq_dist, max_len=max_len, num_layers=NUM_DECODER_LAYERS,
                                             units=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout_rate=DROPOUT)

    # Compile decoder
    transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=masked_loss, metrics=[masked_acc])

    # Create captioner
    captioner = Captioner(features=features, decoder=transformer, tokenizer=tokenizer, max_len=max_len)

    # Train model
    transformer.fit(
        flickr_train_data,
        epochs=EPOCHS,
        validation_data=flickr_valid_data,
    callbacks=[CaptionCallback(valid_files[0], captioner)])
    transformer.save_weights("models/transformer")"""


    # Create LSTM decoder
    # TODO: KeyError with above


    # lstm = LSTMDecoder(max_len=max_len, num_layers=NUM_DECODER_LAYERS, embed_dim=EMBEDDING_DIM,
    #                                          units=EMBEDDING_DIM,
    #                                          vocab_size=vocab_size,
    #                                           dropout_rate=DROPOUT,
    #                                           features=features)
    #
    # You can modify this back into the lstm class
    img_input = tf.keras.Input(shape=(2048,))
    seq_input = tf.keras.Input(shape=(max_len,))
    img_features = tf.keras.layers.Dense(256, activation='relu')(img_input)
    img_features_reshaped = tf.keras.layers.Reshape((1, 256), input_shape=(256,))(img_features)

    seq_features = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=False)(seq_input)
    merged = tf.keras.layers.concatenate([img_features_reshaped, seq_features], axis=1)
    seq_features = tf.keras.layers.LSTM(256)(merged)
    x = tf.keras.layers.Dropout(0.5)(seq_features)
    x = tf.keras.layers.add([x, img_features])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output= tf.keras.layers.Dense(vocab_size)(x) # Removed softmax since its done in caption generation/loss fcn

    lstm = tf.keras.Model(inputs=[img_input, seq_input], outputs = output)

    # Compile decoder
    lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Create captioner
    captioner = Captioner(features=features, decoder=lstm, tokenizer=tokenizer, max_len=max_len)

    # made it here :-)

    # load data: index up to 5 or 10
    # epochs to 1 
    # see how that works

    # TODO garbage below
    """train_generator = CustomDataGenerator(df=flickr_train_data,X_col='image',y_col='caption',batch_size=BATCH_SIZE,directory="flicker8k/Images",
                                      tokenizer=tokenizer,vocab_size=vocab_size,max_length=max_len,features=features)"""

    # Train model
    lstm.fit(
        flickr_train_data,
        epochs=1, # EPOCHS
        validation_data=flickr_valid_data)
    # callbacks=[CaptionCallback(valid_files[0], captioner)])
    lstm.save_weights("models/lstm")
    # TODO: this doesn't work...

    print("poggers")

    # Evaluation: load model and save captions to .txt file
    # lstm.load_weights('models/transformer')
    print(captioner.generate_caption(test_files[0], 1))
    # with open(f'flickr8k/Output/captions_transformer{EPOCHS}.txt', mode='w') as f:
    #     f.write('image,caption\n')
    #     for img in test_files:
    #         caption = captioner.generate_caption(img)
    #         f.write(f'{img},{caption}\n')


    # Evaluation: load model and save captions to .txt file
    # transformer.load_weights('models/transformer')
    # with open(f'flickr8k/Output/captions_transformer.txt', mode='w') as f:
    #     f.write('image,caption\n')
    #     for img in test_files:
    #         caption = captioner.generate_caption(img, 3)
    #         f.write(f'{img},{caption}\n')
