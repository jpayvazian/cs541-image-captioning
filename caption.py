import tensorflow as tf
import numpy as np

class Captioner:
    '''
    Generates image captions via beam search using the trained decoder model
    Returns caption without <start> and <end> tags for easier evaluation with captions_clean_tagless
    '''
    def __init__(self, features, model, tokenizer, max_len, decoder_type):
        self.features = features
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.decoder_type = decoder_type

    def generate_caption(self, image, k):
        '''
        Iterate at most max caption length (-1 for <start>) times:
            For each seq in beam (initially only <start> seq), run seq through model to find top k best next tokens
            Append new tokens to seq and update seq loss, save pairs in temp beam
            After all pairs saved, sort temp beam to keep only top k loss seq to replace beam
            If seq in beam has <end> token, skip running through model and add directly to temp beam
            This way we dont add tokens after <end> and can continue to consider seq without <end> in beam
            Break early if best seq in beam has <end> token
        '''
        features = self.features[image]
        if self.decoder_type == "lstm_attention":
            hidden = self.model.decoder.init_state(batch_size=1)
            features = self.model.encoder(features)

        start = [self.tokenizer.word_index['<start>']]
        end = self.tokenizer.word_index['<end>']
        beam = [(start, 0.0)]

        for _ in range(self.max_len - 1):
            beam_new = []
            for seq, loss in beam:
                if seq[-1] != end:
                    if self.decoder_type == 'transformer':
                        logits = self.model.predict((features, tf.constant([seq])), verbose=0)[:,-1,:]
                    elif self.decoder_type == 'lstm_baseline':
                        logits = self.model.predict((features, tf.keras.utils.pad_sequences([seq], maxlen=self.max_len, padding='post')), verbose=0)
                    elif self.decoder_type == 'lstm_attention':
                        logits, hidden, _ = self.model.decoder.predict((features, tf.constant([seq[-1:]]), hidden))

                    scores = tf.math.log(tf.nn.softmax(logits)).numpy()[0]
                    top_k_idx = np.argsort(scores)[-k:]

                    for idx in top_k_idx:
                        beam_new.append((seq + [idx], loss + scores[idx]))
                else:
                    beam_new.append((seq, loss))

            beam_new.sort(key=lambda x: x[1], reverse=True)
            beam = beam_new[:k]

            if beam[0][0][-1] == end:
                break

        return self.tokenizer.sequences_to_texts([beam[0][0][1:-1]])[0]


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
        print(self.captioner.generate_caption(self.img, k=1))
        print()