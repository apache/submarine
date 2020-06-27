# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


def batch_norm_layer(x, train_phase, scope_bn, batch_norm_decay):
    bn_train = tf.contrib.layers.batch_norm(x,
                                            decay=batch_norm_decay,
                                            center=True,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=True,
                                            reuse=None,
                                            scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x,
                                            decay=batch_norm_decay,
                                            center=True,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=False,
                                            reuse=True,
                                            scope=scope_bn)
    return tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train,
                   lambda: bn_infer)


def dnn_layer(inputs,
              estimator_mode,
              batch_norm,
              deep_layers,
              dropout,
              batch_norm_decay=0.9,
              l2_reg=0,
              **kwargs):
    """
    The Multi Layer Perceptron
    :param inputs: A tensor of at least rank 2 and static value for the last dimension; i.e.
           [batch_size, depth], [None, None, None, channels].
    :param estimator_mode: Standard names for Estimator model modes. `TRAIN`, `EVAL`, `PREDICT`
    :param batch_norm: Whether use BatchNormalization before activation or not.
    :param batch_norm_decay: Decay for the moving average.
           Reasonable values for decay are close to 1.0, typically in the
           multiple-nines range: 0.999, 0.99, 0.9, etc.
    :param deep_layers: list of positive integer, the layer number and units in each layer.
    :param dropout: float in [0,1). Fraction of the units to dropout.
    :param l2_reg: float between 0 and 1.
           L2 regularizer strength applied to the kernel weights matrix.
    """
    with tf.variable_scope("DNN_Layer"):
        if batch_norm:
            if estimator_mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False

        for i in range(len(deep_layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=deep_layers[i],
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                scope='mlp%d' % i)
            if batch_norm:
                deep_inputs = batch_norm_layer(
                    deep_inputs,
                    train_phase=train_phase,
                    scope_bn='bn_%d' % i,
                    batch_norm_decay=batch_norm_decay)
            if estimator_mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        deep_out = tf.contrib.layers.fully_connected(
            inputs=deep_inputs,
            num_outputs=1,
            activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            scope='deep_out')
        deep_out = tf.reshape(deep_out, shape=[-1])
    return deep_out


def linear_layer(features, feature_size, field_size, l2_reg=0, **kwargs):
    """
    Layer which represents linear function.
    :param features: input features
    :param feature_size: size of features
    :param field_size: number of fields in the features
    :param l2_reg: float between 0 and 1.
           L2 regularizer strength applied to the kernel weights matrix.
    """
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    with tf.variable_scope("LinearLayer_Layer"):
        linear_bias = tf.get_variable(name='linear_bias',
                                      shape=[1],
                                      initializer=tf.constant_initializer(0.0))
        linear_weight = tf.get_variable(
            name='linear_weight',
            shape=[feature_size],
            initializer=tf.glorot_normal_initializer(),
            regularizer=regularizer)

        feat_weights = tf.nn.embedding_lookup(linear_weight, feat_ids)
        linear_out = tf.reduce_sum(tf.multiply(feat_weights, feat_vals),
                                   1) + linear_bias
    return linear_out


def embedding_layer(features,
                    feature_size,
                    field_size,
                    embedding_size,
                    l2_reg=0,
                    **kwargs):
    """
    Turns positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    :param features: input features
    :param feature_size: size of features
    :param field_size: number of fields in the features
    :param embedding_size: sparse feature embedding_size
    :param l2_reg: float between 0 and 1.
           L2 regularizer strength applied to the kernel weights matrix.
    """
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    with tf.variable_scope("Embedding_Layer"):
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        embedding_dict = tf.get_variable(
            name='embedding_dict',
            shape=[feature_size, embedding_size],
            initializer=tf.glorot_normal_initializer(),
            regularizer=regularizer)
        embeddings = tf.nn.embedding_lookup(embedding_dict, feat_ids)
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embedding_out = tf.multiply(embeddings, feat_vals)
    return embedding_out


def bilinear_layer(inputs, **kwargs):
    """
    Bi-Interaction Layer used in Neural FM,compress the pairwise element-wise product of features
    into one single vector.
    :param inputs: input features
    """

    with tf.variable_scope("BilinearLayer_Layer"):
        sum_square = tf.square(tf.reduce_sum(inputs, 1))
        square_sum = tf.reduce_sum(tf.square(inputs), 1)
        bilinear_out = 0.5 * tf.subtract(sum_square, square_sum)
    return bilinear_out


def fm_layer(inputs, **kwargs):
    """
    Factorization Machine models pairwise (order-2) feature interactions
    without linear term and bias.
    :param inputs: input features
    """
    with tf.variable_scope("FM_Layer"):
        sum_square = tf.square(tf.reduce_sum(inputs, 1))
        square_sum = tf.reduce_sum(tf.square(inputs), 1)
        fm_out = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)
    return fm_out
