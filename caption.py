import tensorflow as tf

class Captioner:
    '''
    Generates image captions using the trained decoder model
    Returns caption without <start> and <end> tags for easier evaluation with captions_clean_tagless
    Takes greedy approach with temp=0 or sampling approach based on logits if temp>=1
    Larger temp value will introduce more variance in captions
    '''
    def __init__(self, features, decoder, tokenizer, max_len):
        self.features = features
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_len = max_len

    def generate_caption(self, image, temp=0):
        features = self.features[image]
        seq = [self.tokenizer.word_index['<start>']]
        end = self.tokenizer.word_index['<end>']

        for n in range(self.max_len - 2):
            seq_padded = tf.keras.utils.pad_sequences([seq], self.max_len)
            logits = self.decoder.predict((features, seq_padded), verbose=0)[:,-1,:]

            # Greedy/sampling
            if temp == 0:
                yhat = tf.argmax(logits, axis=-1).numpy()[0]
            else:
                yhat = tf.random.categorical(logits/temp, num_samples=1).numpy()[0][0]

            if yhat == end:
                break

            seq.append(yhat)

        return self.tokenizer.sequences_to_texts([seq[1:]])[0]


    # TODO
    def beam_search(self, k):
        '''
        Beam search to keep track of top k captions during generation
        '''
        pass

class CaptionCallback(tf.keras.callbacks.Callback):
    '''
    Callback to generate captions during training for a sample image, to help gauge progress
    Generates with greedy approach and sampling at different temp values for comparison
    '''
    def __init__(self, img, captioner):
        super().__init__()
        self.img = img
        self.captioner = captioner

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        for t in (0, 0.5, 0.75):
            print(self.captioner.generate_caption(self.img, temp=t))