import tensorflow as tf
# adapted from: https://www.tensorflow.org/tutorials/text/image_captioning

class SeqEmbedding(tf.keras.layers.Layer):
    '''
    Input Embedding layer
    Adds embedding vectors for each token and sequence position
    Learns positional embeddings rather than using fixed ones
    '''
    def __init__(self, vocab_size, max_length, embedding_dim):
        '''
        vocab_size: Number unique words (classes) in vocab
        max_length: Biggest caption size (words) to pad sequences to
        embedding dim: Dimensionality of embedding vector
        mask_zero param to init mask of model which ignores 0s as padding tokens
        '''
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=embedding_dim)
        self.token_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq_token = self.token_embedding(seq) #(batch, seq, embedding_dim)

        # Define tensor of positions (0..max_length)
        seq_pos = tf.range(tf.shape(seq_token)[1])[tf.newaxis, :] #(1, seq)
        seq_pos = self.pos_embedding(seq_pos)  #(1, seq, embedding_dim)

        return self.add([seq_token, seq_pos])


class CausalSelfAttention(tf.keras.layers.Layer):
    '''
    MultiHead (Self) Attention layer to attend to the sequence output so far
    Causal (Masked) since we only attend to past positions of sequence
    LayerNormalization: Activation normalization for training stability (similar to batch norm)
    Add: (instead of +) to propogate keras mask
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, seq):
        attn = self.mha(query=seq, value=seq, use_causal_mask=True) #key=value
        seq = self.add([seq, attn])
        return self.layernorm(seq)


class CrossAttention(tf.keras.layers.Layer):
    '''
    CrossAttention layer to attend to different parts of image based on decoded output seq
    "Crosses over" two different context sequences (encoded image features and decoded output seq)
    Unlike CausalSelfAttention which only self-attends to output seq
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, seq, context):
        # Query = output sequence, key/value = image context
        attn, attn_scores = self.mha(query=seq, value=context, return_attention_scores=True)
        # Save attention scores for later visualization
        self.attn_scores = attn_scores
        seq = self.add([seq, attn])
        return self.layernorm(seq)


class FeedForward(tf.keras.layers.Layer):
    '''
    Fully connected layers with ReLU and dropout to further process output seq independently
    '''
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(units=2*units, activation='relu'),
            tf.keras.layers.Dense(units=units),
            tf.keras.layers.Dropout(rate=dropout_rate)
        ])
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.add([seq, self.ff(seq)])
        return self.layernorm(seq)


class DecoderLayer(tf.keras.layers.Layer):
    '''
    Main decoder layer 3 sublayers:
    - CausalSelfAttention
    - CrossAttention
    - FeedForward
    '''
    def __init__(self, units, num_heads, dropout_rate):
        super().__init__()
        self.self_attn = CausalSelfAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.cross_attn = CrossAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def call(self, seq, context):
        seq = self.self_attn(seq)
        seq = self.cross_attn(seq, context)
        self.attn_scores = self.cross_attn.attn_scores

        return self.ff(seq) #(batch, sequence, channels)

# TODO: Output layer
class TokenOutput(tf.keras.layers.Layer):
    '''
    Output layer to generate logit predictions over the vocabulary for each seq position
    '''
    def __init__(self):
        super().__init__()

    def call(self):
        pass

# TODO: Assemble all parts
class TransformerDecoder(tf.keras.Model):
    '''
    Transformer decoder 3 parts:
        - SeqEmbedding
        - DecoderLayer(s)
        - TokenOutput
    '''
    def __init__(self):
        super().__init__()

    def call(self):
        pass
