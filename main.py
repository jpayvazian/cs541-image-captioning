import pandas as pd
import tensorflow as tf
from encoder import extract_features, LSTM_Encoder
from dataset import FlickrDataset
from decoder_transformer import TransformerDecoder
from decoder_lstm_attention import LSTM_Decoder, LSTM_Attention_Model
from decoder_baseline import Decoder_Baseline
from caption import Captioner, CaptionCallback
from utils import get_freq
from eval import masked_loss_transformer, masked_loss_lstm
import numpy as np
import sys

MAX_VOCAB_SIZE = 5000
NUM_DECODER_LAYERS = 2
EMBEDDING_DIM = 256
UNITS = 128
NUM_HEADS = 2
DROPOUT = 0.5
EPOCHS = 10
BATCH_SIZE = 32
LOG_FREQ = 200
ENCODER_TYPES = ['resnet', 'vit']
DECODER_TYPES = ['transformer', 'lstm_baseline', 'lstm_attention']

if __name__ == "__main__":
    '''
    COMMAND LINE ARGS:
    python main.py [ENCODER_TYPE] [DECODER_TYPE]
    '''
    ENCODER_TYPE, DECODER_TYPE = sys.argv[1], sys.argv[2]
    if (ENCODER_TYPE not in ENCODER_TYPES) or (DECODER_TYPE not in DECODER_TYPES):
        print("Invalid encoder/decoder type")
        sys.exit(1)

    # Load data
    labels = pd.read_csv('flickr8k/Labels/captions_clean.csv')
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

    # Feature extraction (Pooling if lstm_baseline)
    features = extract_features(image_files, ENCODER_TYPE)
    if DECODER_TYPE == 'lstm_baseline':
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(train_captions)
        vocab_size = len(tokenizer.word_index) + 1
        features = dict((k, tf.keras.layers.GlobalAveragePooling2D()(np.expand_dims(v, axis=1)).numpy())
                        for k, v in features.items())
    # Tokenizer
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=MAX_VOCAB_SIZE, oov_token="<unk>")
        tokenizer.fit_on_texts(train_captions)
        vocab_size = MAX_VOCAB_SIZE
        freq_dist = get_freq(train_captions, tokenizer.word_index)

    # Max caption size (for padding)
    max_len = max(len(caption.split()) for caption in captions)

    # Create datasets to serve as batch generator during training
    flickr_train_data = FlickrDataset(df=train_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features, decoder_type=DECODER_TYPE)
    flickr_valid_data = FlickrDataset(df=valid_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features, decoder_type=DECODER_TYPE)

    # Create decoder model
    if DECODER_TYPE == "transformer":
        model = TransformerDecoder(freq_dist=freq_dist, max_len=max_len, num_layers=NUM_DECODER_LAYERS,
                                             units=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout_rate=DROPOUT)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=masked_loss_transformer)

    elif DECODER_TYPE == "lstm_attention":
        model = LSTM_Attention_Model(encoder=LSTM_Encoder(EMBEDDING_DIM),
                                    decoder=LSTM_Decoder(freq_dist=freq_dist, embed_dim=EMBEDDING_DIM, units=UNITS),
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                    loss_fcn=masked_loss_lstm,
                                    tokenizer=tokenizer)
    elif DECODER_TYPE == "lstm_baseline":
        model = Decoder_Baseline(encoder=LSTM_Encoder(EMBEDDING_DIM),
                                 units=UNITS, embed_dim=EMBEDDING_DIM, vocab_size=vocab_size, dropout=DROPOUT)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Create captioner
    captioner = Captioner(features=features, model=model, tokenizer=tokenizer, max_len=max_len,
                          decoder_type=DECODER_TYPE)

    # Train model
    if DECODER_TYPE == "transformer":
        model.fit(
            flickr_train_data,
            epochs=EPOCHS,
            validation_data=flickr_valid_data,
            callbacks=[CaptionCallback(valid_files[0], captioner)])

    elif DECODER_TYPE == "lstm_attention":
        for epoch in range(EPOCHS):
            total_train_loss, total_val_loss = 0, 0
            # Training data loop
            for batch, (img_feature, target) in enumerate(flickr_train_data):
                t_loss = model.train_step(img_feature, target)
                total_train_loss += t_loss

                if (batch+1) % LOG_FREQ == 0:
                    print(f'Epoch {epoch+1} Batch {batch+1} Loss {total_train_loss/batch+1:.6f}')

            # Validation data loop
            for batch, (val_img_feature, val_target) in enumerate(flickr_valid_data):
                v_loss = model.test_step(val_img_feature, val_target)
                total_val_loss += v_loss

            avg_train_loss, avg_val_loss = total_train_loss / len(flickr_train_data), total_val_loss / len(flickr_valid_data)
            print(f'Epoch {epoch+1} Training Loss {avg_train_loss:.6f} Valid Loss {avg_val_loss:.6f}')
            print(captioner.generate_caption(valid_files[0], 1))

    elif DECODER_TYPE == "lstm_baseline":
        model.fit(
            flickr_train_data,
            epochs=EPOCHS,
            validation_data=flickr_valid_data,
            callbacks=[CaptionCallback(valid_files[0], captioner)])

    # model.save_weights(f"models/{ENCODER_TYPE}_{DECODER_TYPE}")
    # model.load_weights(f'models/{ENCODER_TYPE}_{DECODER_TYPE}')

    # Evaluation: load model and save captions to .txt file
    with open(f'flickr8k/Output/captions_{ENCODER_TYPE}_{DECODER_TYPE}.txt', mode='w') as f:
        f.write('image,caption\n')
        for img in test_files:
            caption = captioner.generate_caption(img, 3)
            f.write(f'{img},{caption}\n')
