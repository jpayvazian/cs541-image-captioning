import tensorflow as tf

class Captioner:
    '''
    Generates image captions using the trained decoder model
    Returns caption without <start> and <end> tags for easier evaluation with captions_clean_tagless
    '''
    def __init__(self, features, decoder, tokenizer, max_len):
        self.features = features
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_len = max_len

    def generate_caption(self, image):
        features = self.features[image]
        seq = self.tokenizer.texts_to_sequences(['<start>'])
        end = self.tokenizer.texts_to_sequences(['<end>'])[0][0]

        for n in range(self.max_len):
            seq_padded = tf.keras.utils.pad_sequences(seq, self.max_len)

            logits = self.decoder.predict((features, seq_padded), verbose=0)[0,-1,:]
            # TODO: Replace with sampling/beam search
            yhat = tf.argmax(logits).numpy()

            if n == self.max_len - 1:
                yhat = end

            seq[0].append(yhat)

            if yhat == end:
                break

        caption = self.tokenizer.sequences_to_texts(seq)[0]
        return " ".join([word for word in caption.split()[1:-1]])