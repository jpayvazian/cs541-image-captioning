import pandas as pd
import tensorflow as tf
import encoder

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

    # TODO: Encode captions and pad sequence

    # TODO: Feature extraction (run through resnet)
    features = encoder.extract_features(image_files)
