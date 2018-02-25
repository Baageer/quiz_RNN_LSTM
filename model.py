#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='self.keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):

            #hidden_size=256
            stacked_rnn = []
            for iLayer in range(self.rnn_layers):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
                stacked_rnn.append(lstm_cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)

            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            #print('xxxxxxxx', self.X)
            #print('yyyyyyyy', self.Y)
            rnn_inputs = tf.one_hot(self.X, self.num_words)
            #print('rnn_inputs: ', rnn_inputs.shape)

            outputs_tensor, final_state = tf.nn.dynamic_rnn(cell, inputs=rnn_inputs, initial_state=init_state, time_major=False)
            #print(outputs_tensor)
            #print(final_state)
            outputs_state_tensor = outputs_tensor[:, -1, :]
            self.outputs_state_tensor = outputs_state_tensor
            self.state_tensor = outputs_state_tensor

        # concate every time step
        seq_output = tf.concat(outputs_tensor, 1)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])
        #print('seq_output.shape: ', seq_output.shape)
        #print('seq_output_final.shape: ', seq_output_final.shape)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [128, self.num_words], 
                                initializer=tf.random_normal_initializer(stddev=0.01))
            bias = tf.get_variable('b', [self.num_words], initializer=tf.constant_initializer(0.0))
            logits = tf.matmul(seq_output_final, W) + bias
        #print('outputs_state_tensor.shape: ', outputs_state_tensor.shape)
        #print('logits.shape: ', logits.shape)

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        y_one_hot = tf.one_hot(self.Y, self.num_words)
        #print('y_one_hot.shape: ', y_one_hot.shape)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)
        
        self.merged_summary_op = tf.summary.merge_all()
