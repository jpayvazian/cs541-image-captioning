# TODO:
#  - LSTM, soft attention
#  - Transformer (self/cross attention)
#  - Teacher Forcing (with scheduled sampling)?
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.models import Sequential, Model
from keras import initializers, regularizers, constraints

# TODO: fix attention!
def make_model(max_len, vocab_size, embed_dim=256, dropout=0.5, has_attention=False):

    img_input = tf.keras.Input(shape=(2048,))
    seq_input = tf.keras.Input(shape=(max_len,))

    

    img_features = tf.keras.layers.Dense(embed_dim, activation='relu')(img_input)
    img_features_reshaped = tf.keras.layers.Reshape((1, embed_dim), input_shape=(embed_dim,))(img_features)

    seq_features = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)(seq_input)
    merged = tf.keras.layers.concatenate([img_features_reshaped, seq_features], axis=1)

    # TODO fix this. ideas:
    # look @ other notebooks/tutorials
    # play around w this
    # ...
    if has_attention:
        attention = Attention_model(128) # units
        hidden = tf.zeros((32, 128)) # batch size, units
        context_vector, attention_weights = attention(merged, hidden) #TODO: how do i implement
        c = tf.concat([tf.expand_dims(context_vector, 1), merged], axis= -1)
        seq_features = tf.keras.layers.LSTM(embed_dim)(c)
    else:
        seq_features = tf.keras.layers.LSTM(embed_dim)(merged)

    x = tf.keras.layers.Dropout(dropout)(seq_features)
    x = tf.keras.layers.add([x, img_features])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    output= tf.keras.layers.Dense(vocab_size)(x) # Removed softmax since its done in caption generation/loss fcn
    model = tf.keras.Model(inputs=[img_input, seq_input], outputs = output)
    
    return model
    
    
# TODO: again, fix!
class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) 
        self.W2 = tf.keras.layers.Dense(units) 
        self.V = tf.keras.layers.Dense(1) 
        self.units=units

    def call(self, features, hidden):
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1) 
        context_vector = attention_weights * features 
        context_vector = tf.reduce_sum(context_vector, axis=1)  
        return context_vector, attention_weights