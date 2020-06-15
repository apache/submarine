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

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def libsvm_input_fn(
        filepath,
        batch_size=256,
        num_epochs=3,  # pylint: disable=W0613
        perform_shuffle=False,
        delimiter=" ",
        **kwargs):

    def _input_fn():

        def decode_libsvm(line):
            columns = tf.string_split([line], delimiter)
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values, splits.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals,
                                           num_or_size_splits=2,
                                           axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

        dataset = tf.data.TextLineDataset(filepath)\
            .map(decode_libsvm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        return dataset

    return _input_fn
