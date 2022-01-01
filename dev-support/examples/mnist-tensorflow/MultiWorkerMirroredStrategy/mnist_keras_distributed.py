"""
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
"""
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import submarine

BUFFER_SIZE = 10000
BATCH_SIZE = 32

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)

    return (
        datasets["train"]
        .map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
        .shuffle(BUFFER_SIZE)
    )


def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


tf_config = json.loads(os.environ["TF_CONFIG"])
NUM_WORKERS = len(tf_config["cluster"]["worker"])

# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

# Creation of dataset needs to be after MultiWorkerMirroredStrategy object
# is instantiated.
train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)

# next three line is the key point to fix this problem
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)  # AutoShardPolicy.OFF can work too.
train_datasets_no_auto_shard = train_datasets.with_options(options)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()

# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.

# attention:   x=train_datasets_no_auto_shard , not x = train_datasets

if __name__ == "__main__":
    EPOCHS = 5
    hist = multi_worker_model.fit(x=train_datasets_no_auto_shard, epochs=EPOCHS, steps_per_epoch=5)
    for i in range(EPOCHS):
        submarine.log_metric("val_loss", hist.history["loss"][i], i)
        submarine.log_metric("Val_accuracy", hist.history["accuracy"][i], i)


"""Reference
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
https://reurl.cc/no9Zk8
"""
