# +
import torch.nn as nn
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

#tf.enable_eager_execution()
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Embedding,
    Input,
    LSTM,
)
import on_lstm_layer

def get_feature_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get np arrays of upto max_length tokens and person idxs."""
    bet = df.text_between
    #left = df.apply(lambda c: c.tokens[: c.person1_word_idx[0]][-4:-1], axis=1)
    left=df.person1_left_tokens
    right = df.person2_right_tokens


    def pad_or_truncate(l, max_length=40):
        return l[:max_length] + [""] * (max_length - len(l))

    left_tokens = np.array(list(map(pad_or_truncate, left)))
    bet_tokens = np.array(list(map(pad_or_truncate, bet)))
    right_tokens = np.array(list(map(pad_or_truncate, right)))
    # left_tokens = list(map(pad_or_truncate, left))
    # bet_tokens = list(map(pad_or_truncate, bet))
    # right_tokens = list(map(pad_or_truncate, right))
    return left_tokens, bet_tokens, right_tokens


def onlstm(
    tokens: tf.Tensor,
    chunk_sizes=10,
    rnn_state_size: int = 650,
    num_buckets: int = 40000,
    embed_dim: int = 64
):
    ids = tf.strings.to_hash_bucket(tokens, num_buckets)
    embedded_input = Embedding(num_buckets, embed_dim)(ids)
    onlstms=on_lstm_layer.OrderedNeuronLSTM(
                units=rnn_state_size,
                chunk_size=chunk_sizes,
                dropout=0.4
                )
    return onlstms(embedded_input)



def get_model(
     embed_dim: int = 36
) -> tf.keras.Model:
    """
    Return LSTM model for predicting label probabilities.
    Args:
        rnn_state_size: LSTM state size.
        num_buckets: Number of buckets to hash strings to integers.
        embed_dim: Size of token embeddings.
    Returns:
        model: A compiled LSTM model.
    """
    left_ph = Input((None,), dtype="string")
    bet_ph = Input((None,), dtype="string")
    right_ph = Input((None,), dtype="string")
    left_embs = onlstm(left_ph,embed_dim=embed_dim)
    bet_embs = onlstm(bet_ph,embed_dim=embed_dim)
    right_embs = onlstm(right_ph, embed_dim=embed_dim)
    layer = Concatenate(1)([left_embs, bet_embs, right_embs])
    layer = Dense(64, activation=tf.nn.relu)(layer)
    layer = Dense(32, activation=tf.nn.relu)(layer)
    probabilities = Dense(2, activation=tf.nn.softmax)(layer)
    model = tf.keras.Model(inputs=[bet_ph, left_ph, right_ph], outputs=probabilities)
    model.compile(tf.train.AdagradOptimizer(0.1), "categorical_crossentropy")
    return model


