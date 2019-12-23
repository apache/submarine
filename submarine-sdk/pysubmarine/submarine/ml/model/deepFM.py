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
import submarine.ml.utils
from submarine.ml.parameters import defaultParameters
from submarine.ml.utils import get_TFConfig, get_from_registry, sanity_checks
from submarine.ml.model.abstractModel import abstractModel
from submarine.ml.registries import input_fn_registry

logger = logging.getLogger(__name__)


def batch_norm_layer(x, train_phase, scope_bn, batch_norm_decay):
    bn_train = tf.contrib.layers.batch_norm(x, decay=batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True,
                                            reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False,
                                            reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def model_fn(features, labels, mode, params):
    field_size = params["training"]["field_size"]
    feature_size = params["training"]["feature_size"]
    embedding_size = params["training"]["embedding_size"]
    l2_reg = params["training"]["l2_reg"]
    learning_rate = params["training"]["learning_rate"]
    batch_norm = params["training"]["batch_norm"]
    batch_norm_decay = params["training"]["batch_norm_decay"]
    optimizer = params["training"]["optimizer"]
    seed = params["training"]["seed"]
    metric = params['output']['metric']

    layers = list(map(int, params["training"]["deep_layers"].split(',')))
    dropout = list(map(float, params["training"]["dropout"].split(',')))

    tf.set_random_seed(seed)
    fm_bias = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
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

    with tf.variable_scope("Deep-part"):
        if batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False

        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i],
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                scope='mlp%d' % i)
            if batch_norm:
                deep_inputs = batch_norm_layer(
                    deep_inputs, train_phase=train_phase,
                    scope_bn='bn_%d' % i, batch_norm_decay=batch_norm_decay)
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        y_deep = tf.contrib.layers.fully_connected(
            inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            scope='deep_out')
        y_d = tf.reshape(y_deep, shape=[-1])

    with tf.variable_scope("DeepFM-out"):
        y_bias = fm_bias * tf.ones_like(y_d, dtype=tf.float32)
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions = {"probabilities": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
               l2_reg * tf.nn.l2_loss(fm_weight) + l2_reg * tf.nn.l2_loss(fm_vector)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {}
    if metric == 'auc':
        eval_metric_ops['auc'] = tf.metrics.auc(labels, pred)
    else:
        raise TypeError("Can not find loss_type :", params['training']['loss_type'])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    with tf.name_scope("Optimizer"):
        if optimizer == 'adam':
            op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif optimizer == 'adagrad':
            op = tf.train.AdagradOptimizer(
                learning_rate=learning_rate, initial_accumulator_value=1e-8)
        elif optimizer == 'momentum':
            op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
        elif optimizer == 'ftrl':
            op = tf.train.FtrlOptimizer(learning_rate)
        else:
            raise TypeError("Can not find optimizer :", optimizer)

    train_op = op.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


class deepFM(abstractModel):
    def __init__(self, model_params=None, json_path=None):
        """
        Create a tensorflow DeepFM model
        :param model_params: defines the different
        parameters of the model, features, preprocessing and training
        :param json_path: The JSON file that specifies the model parameters.
        """
        super().__init__()
        self.model_params = submarine.ml.utils.merge_dicts(model_params, defaultParameters)
        self.model_params = submarine.ml.utils.merge_json(json_path, self.model_params)
        sanity_checks(self.model_params)
        logging.info("Model parameters : %s", self.model_params)
        self.input_type = self.model_params['input']['type']
        self.model_dir = self.model_params['output']['save_model_dir']
        self.config = get_TFConfig(self.model_params)
        self.model = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=self.model_dir,
            params=self.model_params, config=self.config)

    def train(self, train_input_fn=None, eval_input_fn=None, **kwargs):
        """
        Trains a pre-defined tensorflow estimator model with given training data
        :param train_input_fn: A function that provides input data for training.
        :param eval_input_fn: A function that provides input data for evaluating.
        :return: None
        """
        if train_input_fn is None:
            train_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                filepath=self.model_params['input']['train_data'],
                **self.model_params['training'])
        if eval_input_fn is None:
            eval_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                filepath=self.model_params['input']['valid_data'],
                **self.model_params['training'])

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)

    def evaluate(self, eval_input_fn=None, **kwargs):
        """
        Evaluates a pre-defined Tensorflow estimator model with given evaluate data
        :param eval_input_fn: A function that provides input data for evaluating.
        :return: A dict containing the evaluation metrics specified in `eval_input_fn` keyed by
        name, as well as an entry `global_step` which contains the value of the
        global step for which this evaluation was performed
        """
        if eval_input_fn is None:
            eval_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                filepath=self.model_params['input']['valid_data'],
                **self.model_params['training'])

        return self.model.evaluate(input_fn=eval_input_fn, **kwargs)

    def predict(self, test_input_fn=None, **kwargs):
        """
        Yields predictions with given features.
        :param test_input_fn: A function that constructs the features.
         Prediction continues until input_fn raises an end-of-input exception
        :return: Evaluated values of predictions tensors.
        """
        if test_input_fn is None:
            test_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                filepath=self.model_params['input']['test_data'],
                **self.model_params['training'])

        return self.model.predict(input_fn=test_input_fn, **kwargs)
