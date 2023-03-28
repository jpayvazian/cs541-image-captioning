import pandas as pd
import tensorflow as tf
from encoder import extract_features
from dataset import FlickrDataset

if __name__ == "__main__":
    # Load data
    labels = pd.read_csv('flickr8k/captions_clean.csv')
    captions = labels['caption'].tolist()
    image_files = labels['image'].unique().tolist()

    # Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(captions)
    # Number unique words in vocab (for num classes), biggest caption size (for padding)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(caption.split()) for caption in captions)

    # Train/test split
    split_idx = int(len(image_files) * 0.8)
    train_files, test_files = image_files[:split_idx], image_files[split_idx:]

    # Create separate df for train/test labels
    train_labels = labels[labels['image'].isin(train_files)]
    test_labels = labels[labels['image'].isin(test_files)]
    train_labels.reset_index(drop=True)
    test_labels.reset_index(drop=True)

    # Feature extraction (run through resnet)
    features = extract_features(image_files)

    # Create dataset to serve as batch generator during training
    flickr_data = FlickrDataset(df=train_labels, tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_len,
                                batch_size=64, features=features)
