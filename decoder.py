# TODO:
#  - LSTM, soft attention
#  - Transformer (self/cross attention)
#  - Teacher Forcing (with scheduled sampling)?
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.models import Sequential, Model
from keras import initializers, regularizers, constraints


# TODO: make a new one
# basically the same as the tutorial except 
#1: we can fine-tune it
#2: it includes attention. can use attention stuff Jack used/find stuff on google!

def baseline(vocab_size, max_len):

    input1 = Input(shape=(1920,)) # TODO might not be here
    input2 = Input(shape=(max_len,)) # TODO might not be here

    img_features = Dense(256, activation='relu')(input1)
    img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)
    print("reshapped! but was that correct? TODO: mess around and find out :)") # No error message yet it fails!

    # TODO copy me from tutorial
    # https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstms#Modelling

    sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2) # 

    print("we made it here!") # good ig
    #exit(0)

    merged = concatenate([img_features_reshaped,sentence_features],axis=1)
    sentence_features = LSTM(256)(merged)
    x = Dropout(0.5)(sentence_features)
    x = add([x, img_features])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    caption_model = Model(inputs=[input1,input2], outputs=output)
    caption_model.compile(loss='categorical_crossentropy',optimizer='adam')

    # this is using the kaggle model but we can change things up. just get this working first and go from there!

    print(" the end ! ")
    
# TODO: fine-tune and try to get this stuff working.
# also identify areas that can (likely) be improved
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
class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units) #iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer
        

    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed = self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis = -1) # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        
        return output, state, attention_weights
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

def attention_LSTM(features, vocab_size, train_labels, tokenizer):
    sample_cap_batch = train_labels
    embedding_dim = 256  # TODO is this correct?
    units = 512 # TODO: is this correct???
    decoder=Decoder(embedding_dim, units, vocab_size) # TODO: test this
    # idt this decoder even has an LSTM layer... we can worry about that later! for now, just see if we can at least get this code to RUN


    hidden = decoder.init_state(batch_size=sample_cap_batch.shape[0])
    # dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * sample_cap_batch.shape[0], 1)
    dec_input = tf.expand_dims([0] * sample_cap_batch.shape[0], 1)

    predictions, hidden_out, attention_weights= decoder(dec_input, features, hidden)
    print('Feature shape from Encoder: {}'.format(features.shape)) #(batch, 8*8, embed_dim)
    print('Predcitions shape from Decoder: {}'.format(predictions.shape)) #(batch,vocab_size)
    print('Attention weights shape from Decoder: {}'.format(attention_weights.shape)) #(batch, 8*8, embed_dim)