import tensorflow as tf

@tf.function
def masked_loss(y, yhat):
    '''
    Custom cross entropy loss which uses mask to exclude pad/<start> tokens in calculation
    Sparse Categorical since labels are integer encodings (not 1-hot)
    '''
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y, yhat)
    mask = (y != 0) & (loss < 1e8)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

@tf.function
def masked_acc(y, yhat):
    '''
    Custom accuracy which uses mask to exclude pad/<start> tokens in calculation
    Checks equality between most probable next token and label
    '''
    mask = tf.cast(y != 0, tf.float32)
    yhat = tf.argmax(yhat, axis=-1)
    y = tf.cast(y, yhat.dtype)
    correct = tf.cast(yhat == y, mask.dtype)
    return tf.reduce_sum(correct * mask)/tf.reduce_sum(mask)
