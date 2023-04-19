import tensorflow as tf
from decoder_transformer import TokenOutput

class LSTM_Decoder(tf.keras.Model):
    '''
    LSTM RNN with Attention
    '''
    def __init__(self, freq_dist, embed_dim, units):
        super().__init__()
        self.units = units
        self.attention = Attention(units)
        self.embed = tf.keras.layers.Embedding(len(freq_dist), embed_dim)
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.d1 = tf.keras.layers.Dense(units)
        self.d2 = TokenOutput(freq_dist)
        self.d2.smart_init()

    def call(self, inputs):
        features, x, hidden = inputs
        context_vector, attn_weights = self.attention(features, hidden)
        x = self.embed(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, _ = self.lstm(x)
        x = self.d1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.d2(x)

        return x, state_h, attn_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class Attention(tf.keras.Model):
    '''
    Soft (Bahdanau) Attention for focusing on relevant parts of image during caption generation
    '''
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class LSTM_Attention_Model:
    '''
    Contains both LSTM encoder/decoder to perform custom training step
    '''
    def __init__(self, encoder, decoder, optimizer, loss_fcn, tokenizer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_fcn = loss_fcn
        self.tokenizer = tokenizer

    @tf.function
    def train_step(self, img_feature, target):
        loss = 0
        hidden = self.decoder.init_state(batch_size=target.shape[0])
        seq = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)
        with tf.GradientTape() as tape:
            features = self.encoder(img_feature)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder((features, seq, hidden))
                loss += self.loss_fcn(target[:, i], predictions)
                # teacher forcing
                seq = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss

    def test_step(self, img_feature, target):
        loss = 0
        hidden = self.decoder.init_state(batch_size=target.shape[0])
        seq = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)
        features = self.encoder(img_feature)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = self.decoder.predict((features, seq, hidden), verbose=0)
            loss += self.loss_fcn(target[:, i], predictions)
            # use predicted token for next time step
            seq = tf.expand_dims(tf.argmax(predictions, axis=1), 1)

        return loss / int(target.shape[1])