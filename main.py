import pandas as pd
import tensorflow as tf
from encoder import extract_features
from dataset import FlickrDataset
from decoder_transformer import TransformerDecoder
from caption import Captioner, CaptionCallback
from utils import get_freq
from eval import masked_loss, masked_acc

NUM_DECODER_LAYERS = 2
EMBEDDING_DIM = 256
NUM_HEADS = 2
DROPOUT = 0.5
EPOCHS = 10
BATCH_SIZE = 32

if __name__ == "__main__":
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

    # Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(train_captions)
    # Number unique words in vocab (for num classes), biggest caption size (for padding)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(caption.split()) for caption in captions)

    # Feature extraction (run through resnet)
    features = extract_features(image_files, 'vit')

    # Create datasets to serve as batch generator during training
    flickr_train_data = FlickrDataset(df=train_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features)
    flickr_valid_data = FlickrDataset(df=valid_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len,
                                batch_size=BATCH_SIZE, features=features)

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
    transformer.save_weights("models/transformer")

    # Evaluation: load model and save captions to .txt file
    # transformer.load_weights('models/transformer')
    # with open(f'flickr8k/Output/captions_transformer.txt', mode='w') as f:
    #     f.write('image,caption\n')
    #     for img in test_files:
    #         caption = captioner.generate_caption(img, 3)
    #         f.write(f'{img},{caption}\n')
