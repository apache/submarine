# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 64
BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4


def get_task_name():
    cluster_spec = os.environ.get("CLUSTER_SPEC", None)
    task_name = ""
    if cluster_spec:
        cluster_spec = json.loads(cluster_spec)
        job_index = os.environ["TASK_INDEX"]
        job_name = os.environ["JOB_NAME"]
        task_name = job_name + "_" + job_index

    return task_name


def input_fn(mode, input_context=None):
    datasets, info = tfds.load(
        name="mnist",
        data_dir="/tmp/" + get_task_name() + "/data",
        with_info=True,
        as_supervised=True,
    )

    mnist_dataset = datasets["train"] if mode == tf.estimator.ModeKeys.TRAIN else datasets["test"]

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )

    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def model_fn(features, labels, mode):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(labels, logits)
    loss = tf.reduce_sum(loss) * (1.0 / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()),
    )


if __name__ == "__main__":
    strategy = tf.distribute.experimental.ParameterServerStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="/tmp/model", config=config)
    train_spec = tf.estimator.TrainSpec(input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
