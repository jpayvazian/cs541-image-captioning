# TODO:
#  - LSTM, soft attention
#  - Transformer (self/cross attention)
#  - Teacher Forcing (with scheduled sampling)?

from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.models import Sequential, Model



def LSTM(vocab_size, max_len, img_features):

    input1 = Input(shape=(1920,)) # TODO might not be here
    input2 = Input(shape=(max_len,)) # TODO might not be here

    print("we made it here!")
    exit(0)

    img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

    # TODO copy me from tutorial
    # https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstms#Modelling

    sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2) # TODO: figure out the embedding shit

    

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
    

"""

input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)

# below is stuff that I think will be for the decoder. this follows the kaggle example
# so if i wanted to do smth different this may not work...

merged = concatenate([img_features_reshaped,sentence_features],axis=1)
sentence_features = LSTM(256)(merged)
x = Dropout(0.5)(sentence_features)
x = add([x, img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1,input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy',optimizer='adam')

"""