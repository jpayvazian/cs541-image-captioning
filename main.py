import pandas as pd
import tensorflow as tf

TRAIN_SIZE = 0.8
IMG_SIZE = 224

if __name__ == "__main__":
    # Load data
    labels = pd.read_csv('flickr8k/captions_clean.csv')
    captions = labels['caption'].tolist()
    image_files = labels['image'].unique().tolist()

    # Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(captions)
    # Number unique words in vocab, biggest caption size
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(caption.split()) for caption in captions)

    # Train/test split
    split_idx = int(len(image_files) * TRAIN_SIZE)
    train_files, test_files = image_files[:split_idx], image_files[split_idx:]

    # Create separate df for train/test labels
    train_labels = labels[labels['image'].isin(train_files)]
    test_labels = labels[labels['image'].isin(test_files)]
    train_labels.reset_index(drop=True)
    test_labels.reset_index(drop=True)

    # TODO: Feature extraction
    # images = tf.keras.utils.load_img('flickr8k/Images', target_size=(IMG_SIZE,IMG_SIZE))