import tensorflow as tf
# adapted from: https://www.tensorflow.org/tutorials/text/image_captioning

class SeqEmbedding(tf.keras.layers.Layer):
    '''
    Input Embedding layer
    Adds embedding vectors for each token and sequence position
    Learns positional embeddings rather than using fixed ones
    '''
    def __init__(self, vocab_size, max_len, embedding_dim):
        '''
        vocab_size: Number unique words (classes) in vocab
        max_len: Biggest caption size (words) to pad sequences to
        embedding dim: Dimensionality of embedding vector
        mask_zero param to init mask of model which ignores 0s as padding tokens
        '''
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)
        self.token_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq_token = self.token_embedding(seq) #(batch, seq, embedding_dim)

        # Define tensor of positions (0..max_len)
        seq_pos = tf.range(tf.shape(seq_token)[1])[tf.newaxis, :] #(1, seq)
        seq_pos = self.pos_embedding(seq_pos)  #(1, seq, embedding_dim)
        seq_pos = tf.tile(seq_pos, [tf.shape(seq_token)[0], 1, 1]) #(batch, seq, embedding_dim)

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

    def call(self, context, seq):
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
            tf.keras.layers.Dense(2*units, activation='relu'),
            tf.keras.layers.Dense(units),
            tf.keras.layers.Dropout(dropout_rate)
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
        self.ff = FeedForward(units, dropout_rate)

    def call(self, context, seq):
        seq = self.self_attn(seq)
        seq = self.cross_attn(context, seq)
        self.attn_scores = self.cross_attn.attn_scores

        return self.ff(seq) #(batch, sequence, channels)

# TODO: Improve performance
#  Set large negative bias for <start> and <pad> tokens
#  Smart initialization (token dist not uniform)
class TokenOutput(tf.keras.layers.Layer):
    '''
    Fully connected output layer to generate logits of each token
    For numerical stability, Softmax activation is omitted
    from_logits=True used in loss fcn instead
    Reshape layer to expand the logits dim to match labels
    '''
    def __init__(self, vocab_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, seq):
        return self.dense(seq)


class TransformerDecoder(tf.keras.Model):
    '''
    Transformer decoder 3 parts:
        - SeqEmbedding
        - DecoderLayer(s)
        - TokenOutput
    '''
    def __init__(self, vocab_size, max_len, num_layers, units, num_heads, dropout_rate):
        super().__init__()
        self.input_layer = SeqEmbedding(vocab_size, max_len, units)
        self.decoder_layers = [DecoderLayer(units, num_heads, dropout_rate) for n in range(num_layers)]
        self.output_layer = TokenOutput(vocab_size)

    # TODO: Clear keras mask?
    def call(self, inputs):
        context, seq = inputs
        seq = self.input_layer(seq)
        for decoder_layer in self.decoder_layers:
            seq = decoder_layer(context, seq)

        return self.output_layer(seq)
