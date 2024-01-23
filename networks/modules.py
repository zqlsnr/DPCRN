# -*- coding: utf-8 -*-
"""
#!/usr/bin/python3
#-*- coding: utf-8 -*-
@FileName: real_time_tflite.py
@Time: 2022/10/19 13:44
@Author: zql
"""
import tensorflow.keras as keras
from tensorflow.keras.layers import Add
import tensorflow as tf


class DprnnBlock(keras.layers.Layer):

    def __init__(self, intra_hidden, inter_hidden, batch_size, L, width, channel, causal=False, CUDNN=False, **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        '''
        intra_hidden hidden size of the intra-chunk RNN
        inter_hidden hidden size of the inter-chunk RNN
        batch_size 
        L         number of frames, -1 for undefined length
        width     width size output from encoder
        channel   channel size output from encoder
        causal    instant Layer Norm or global Layer Norm
        '''
        self.batch_size = batch_size
        self.causal = causal
        self.L = L
        self.width = width
        self.channel = channel

        if CUDNN:
            self.intra_rnn = keras.layers.Bidirectional(
                keras.layers.CuDNNGRU(units=intra_hidden // 2, return_sequences=True))
        else:
            self.intra_rnn = keras.layers.Bidirectional(
                keras.layers.GRU(units=intra_hidden // 2, return_sequences=True, implementation=1,
                                 recurrent_activation='sigmoid', unroll=True, reset_after=False))

        self.intra_fc = keras.layers.Dense(units=self.channel, )

        self.intra_ln = keras.layers.LayerNormalization(center=True, scale=True, epsilon=1e-8)

        if CUDNN:
            self.inter_rnn = keras.layers.CuDNNGRU(units=inter_hidden, return_sequences=True)
        else:
            self.inter_rnn = keras.layers.GRU(units=inter_hidden, return_sequences=True, implementation=1,
                                              recurrent_activation='sigmoid', reset_after=False)

        self.inter_fc = keras.layers.Dense(units=self.channel, )

        self.inter_ln = keras.layers.LayerNormalization(center=True, scale=True, epsilon=1e-8)

    def call(self, x):
        # Intra-Chunk Processing
        batch_size = self.batch_size
        L = self.L
        width = self.width

        intra_rnn = self.intra_rnn
        intra_fc = self.intra_fc
        intra_ln = self.intra_ln
        inter_rnn = self.inter_rnn
        inter_fc = self.inter_fc
        inter_ln = self.inter_ln
        channel = self.channel
        causal = self.causal
        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_GRU_input = tf.reshape(x, [-1, width, channel])
        # (bs*T,F,C)
        intra_GRU_out = intra_rnn(intra_GRU_input)

        # (bs*T,F,C) channel axis dense
        intra_dense_out = intra_fc(intra_GRU_out)

        if causal:
            # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
            intra_ln_input = tf.reshape(intra_dense_out, [batch_size, -1, width, channel])
            intra_out = intra_ln(intra_ln_input)
        else:
            # (bs*T,F,C) --> (bs,T*F*C) global norm
            intra_ln_input = tf.reshape(intra_dense_out, [batch_size, -1])
            intra_ln_out = intra_ln(intra_ln_input)
            intra_out = tf.reshape(intra_ln_out, [batch_size, L, width, channel])

        # (bs,T,F,C)
        intra_out = Add()([x, intra_out])
        # %%
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_GRU_input = tf.transpose(intra_out, [0, 2, 1, 3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_GRU_input = tf.reshape(inter_GRU_input, [batch_size * width, L, channel])

        inter_GRU_out = inter_rnn(inter_GRU_input)

        # (bs,F,T,C) Channel axis dense
        inter_dense_out = inter_fc(inter_GRU_out)

        inter_dense_out = tf.reshape(inter_dense_out, [batch_size, width, L, channel])

        if causal:
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_ln_input = tf.transpose(inter_dense_out, [0, 2, 1, 3])
            inter_out = inter_ln(inter_ln_input)
        else:
            # (bs,F,T,C) --> (bs,F*T*C)
            inter_ln_input = tf.reshape(inter_dense_out, [batch_size, -1])
            inter_ln_out = inter_ln(inter_ln_input)
            inter_out = tf.reshape(inter_ln_out, [batch_size, width, L, channel])
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_out = tf.transpose(inter_out, [0, 2, 1, 3])

        inter_out = Add()([intra_out, inter_out])

        return inter_out
