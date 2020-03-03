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

"""
TensorFlow implementation of FM

Reference:
[1] Factorization machines for CTR Prediction,
    Steffen Rendle
"""

import logging
import tensorflow as tf
import numpy as np
from submarine.ml.model.base_tf_model import BaseTFModel
from submarine.utils.tf_utils import get_estimator_spec

logger = logging.getLogger(__name__)


class FM(BaseTFModel):
    def model_fn(self, features, labels, mode, params):
        field_size = params["training"]["field_size"]
        feature_size = params["training"]["feature_size"]
        embedding_size = params["training"]["embedding_size"]
        seed = params["training"]["seed"]

        np.random.seed(seed)
        tf.set_random_seed(seed)

        fm_bias = tf.get_variable(name='fm_bias', shape=[1],
                                  initializer=tf.constant_initializer(0.0))
        fm_weight = tf.get_variable(name='fm_weight', shape=[feature_size],
                                    initializer=tf.glorot_normal_initializer())
        fm_vector = tf.get_variable(name='fm_vector', shape=[feature_size, embedding_size],
                                    initializer=tf.glorot_normal_initializer())

        with tf.variable_scope("Feature"):
            feat_ids = features['feat_ids']
            feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
            feat_vals = features['feat_vals']
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

        with tf.variable_scope("First_order"):
            feat_weights = tf.nn.embedding_lookup(fm_weight, feat_ids)
            y_w = tf.reduce_sum(tf.multiply(feat_weights, feat_vals), 1)

        with tf.variable_scope("Second_order"):
            embeddings = tf.nn.embedding_lookup(fm_vector, feat_ids)
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embeddings, feat_vals)
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)

        y = fm_bias + y_w + y_v

        return get_estimator_spec(y, labels, mode, params, [fm_vector, fm_weight])
