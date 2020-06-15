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


class OptimizerKey(object):
    """Optimizer key strings."""
    ADAM = 'adam'
    ADAGRAD = 'adagrad'
    MOMENTUM = 'momentum'
    FTRL = 'ftrl'


def get_optimizer(optimizer_key, learning_rate):
    optimizer_key = optimizer_key.lower()

    if optimizer_key == OptimizerKey.ADAM:
        op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-8)
    elif optimizer_key == OptimizerKey.ADAGRAD:
        op = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                       initial_accumulator_value=1e-8)
    elif optimizer_key == OptimizerKey.MOMENTUM:
        op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                        momentum=0.95)
    elif optimizer_key == OptimizerKey.FTRL:
        op = tf.train.FtrlOptimizer(learning_rate)
    else:
        raise ValueError("Invalid optimizer_key :", optimizer_key)
    return op
