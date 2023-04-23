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
def make_model(inputs, max_len, vocab_size, embed_dim=256, dropout=0.5, has_attention=False, hidden=None):

    

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
        # TODO: idt this is what hidden should be chief
        context_vector, attention_weights = attention(inputs, hidden) #TODO: how do i implement
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
# main tutorial: https://www.kaggle.com/code/lvishalraju/image-captioning-attention
# similar: https://www.kaggle.com/code/trngvhong/image-captioning-attention-flickr8k
class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) 
        self.W2 = tf.keras.layers.Dense(units) 
        self.V = tf.keras.layers.Dense(1) 
        self.units=units

    def call(self, inputs, hidden):
        # features, seq = inputs
        features = inputs
        features = tf.keras.layers.Dense(256)(features)

        # TODO: we are getting the big print-out error here!
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) # This is where the error occurs. No shot it isn't the "features"
        # see how the tutorial has the features setup... and see how ours compare!
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1) 
        context_vector = attention_weights * features 
        context_vector = tf.reduce_sum(context_vector, axis=1)  

        return context_vector, attention_weights
    
class Decoder_Baseline(Model):
    def __init__(self, units, max_len, embed_dim, vocab_size, dropout, has_attention):
        super(Decoder_Baseline, self).__init__()
        self.units = units
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size 
        self.dropout = dropout
        self.has_attention = has_attention
        self.hidden = tf.zeros((32, self.units)) # TODO we can do better than this

        self.attention = Attention_model(self.units)

        self.dense1 = tf.keras.layers.Dense(self.embed_dim, activation='relu')
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim, mask_zero=False)
        self.lstm = tf.keras.layers.LSTM(self.embed_dim)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dense2 = tf.keras.layers.Dense(self.units, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)
        self.dense3 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs):
        img_features, seq_input = inputs

        img_features = self.dense1(img_features)
        img_features_reshaped = tf.keras.layers.Reshape((1, self.embed_dim), input_shape=(self.embed_dim,))(img_features)

        seq_features = self.embedding(seq_input)
        merged = tf.keras.layers.concatenate([img_features_reshaped, seq_features], axis=1)


        # TODO fix this. ideas:
        # look @ other notebooks/tutorials
        # play around w this
        # ...
        if self.has_attention:
            # TODO: idt this is what hidden should be chief
            #print(merged.shape) # the fuck is this shape. idt we want to use merged!
            #print(img_features_reshaped.shape)
            #print(seq_features.shape)
            context_vector, attention_weights = self.attention(inputs, self.hidden) #TODO: how do i implement
            print("SHAPES")
            print(img_features_reshaped.shape)
            print(context_vector.shape)
            c = tf.concat([tf.expand_dims(context_vector, 1), img_features_reshaped], axis= 0) # TODO: error here with dimensions! 
            #print(c.shape)
            #print(seq_features.shape)
            # exit(0)
            # for now :| ignore sequences i guess TODO fix that~
            #merged = tf.keras.layers.concatenate([c, seq_features], axis=1)
            seq_features = tf.keras.layers.LSTM(self.embed_dim)(c)
            
        else:
            seq_features = self.lstm(merged)

        x = self.dropout1(seq_features)
        x = tf.keras.layers.add([x, img_features]) # TODO issue here w/ different batch sizes :/
        x = self.dense2(x)
        x = self.dropout2(x)
        output = self.dense3(x) # Removed softmax since its done in caption generation/loss fcn
        return output


class Decoder_Attention(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder_Attention, self).__init__()
        self.units=units
        self.attention = tf.keras.layers.Attention() # TODO
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer
        self.hidden = self.init_state(32) # TODO garbage
        self.x = None

    def call(self,inputs):
        hidden = self.hidden

        x = self.x
        features = inputs

        context_vector = self.attention([x, features]) #create your context vector & attention weights from attention model
        embed = self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis = -1) # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)

        #(64, 1, 49, 256)
        #(64, 1, 256)

        # is what the dimensions should be....
        # so 32,1,256 is good
        # 360, 34, 256 is NOT good!

        # TODO: error here:
        # [32, 1, 256] vs. [360, 34, 256]

        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        
        return output
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
