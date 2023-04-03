import pandas as pd
import tensorflow as tf
import os
from encoder import extract_features
from dataset import FlickrDataset
from decoder_transformer import TransformerDecoder
from caption import Captioner
from utils import masked_loss

# TODO: optimize hyperparams
NUM_DECODER_LAYERS = 1
EMBEDDING_DIM = 128
NUM_HEADS = 2
DROPOUT = 0.1
EPOCHS = 1
BATCH_SIZE = 32

if __name__ == "__main__":
    # Load data
    '''
    # Since theres 8000img * 5caption * len(max seq), memory/time too big
    # - Take only (longest) caption per image? Easier to compare metrics that way also instead of averageing between 5
    # - Or set num steps per epoch to fixed length?
    '''
    labels = pd.read_csv('flickr8k/captions_clean.csv')[:1000]
    captions = labels['caption'].tolist()
    image_files = labels['image'].unique().tolist()

    # Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(captions)
    # Number unique words in vocab (for num classes), biggest caption size (for padding)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(caption.split()) for caption in captions)

    # Train/test/valid split 70/20/10
    train_idx, test_idx = int(len(image_files) * 0.7), int(len(image_files) * 0.9)
    train_files, test_files, valid_files = image_files[:train_idx], image_files[train_idx:test_idx], image_files[test_idx:]

    # Create separate df for train/test/valid labels
    train_labels = labels[labels['image'].isin(train_files)]
    test_labels = labels[labels['image'].isin(test_files)]
    valid_labels = labels[labels['image'].isin(valid_files)]
    train_labels.reset_index(drop=True)
    test_labels.reset_index(drop=True)
    valid_labels.reset_index(drop=True)

    # Feature extraction (run through resnet)
    features = extract_features(image_files)

    # Create datasets to serve as batch generator during training
    flickr_train_data = FlickrDataset(df=train_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features)
    flickr_valid_data = FlickrDataset(df=valid_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features)

    # Create transformer decoder
    transformer = TransformerDecoder(vocab_size=vocab_size, max_len=max_len, num_layers=NUM_DECODER_LAYERS,
                                             units=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout_rate=DROPOUT)

    # TODO: Optimize hyperparam/lr schedule

    # Compile decoder
    transformer.compile(loss=masked_loss, optimizer='adam')

    # Save model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('models', 'transformer', 'checkpoint-{epoch}'),
        save_weights_only = True
    )

    # Train decoder with checkpoint
    if os.path.isdir("models/transformer"):
        transformer.load_weights(tf.train.latest_checkpoint('models/transformer')).expect_partial() # suppress warnings

    transformer.fit(
        flickr_train_data,
        epochs=EPOCHS,
        validation_data=flickr_valid_data,
        callbacks=[checkpoint]
    )

    # Generate captions for each image
    captioner = Captioner(features=features, decoder=transformer, tokenizer=tokenizer, max_len=max_len)

    # Save captions to .txt file
    with open('flickr8k/captions_yhat.txt', mode='w') as f:
        f.write('image,caption\n')
        for img in test_files[:5]:
            caption = captioner.generate_caption(img)
            f.write(f'{img},{caption}\n')

    #TODO: Evaluate captions with metrics