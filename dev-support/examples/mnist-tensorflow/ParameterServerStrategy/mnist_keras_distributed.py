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

import submarine

print(tf.__version__)

TF_CONFIG = os.environ.get("TF_CONFIG", "")
NUM_PS = len(json.loads(TF_CONFIG)["cluster"]["ps"])
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
    min_shard_bytes=(256 << 10), max_shards=NUM_PS
)

strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver, variable_partitioner=variable_partitioner
)


def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    x = tf.random.uniform((10, 10))
    y = tf.random.uniform((10,))

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    return dataset


dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

model.compile(tf.keras.optimizers.SGD(), loss="mse", steps_per_execution=10)

working_dir = "/tmp/my_working_dir"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
]

# Define the checkpoint directory to store the checkpoints.
checkpoint_dir = "./training_checkpoints"

model.fit(dc, epochs=5, steps_per_epoch=20, callbacks=callbacks)
if __name__ == "__main__":
    EPOCHS = 5
    hist = model.fit(dc, epochs=EPOCHS, steps_per_epoch=20, callbacks=callbacks)
    for i in range(EPOCHS):
        submarine.log_metric("val_loss", hist.history["loss"][i], i)
        submarine.log_metric("Val_accuracy", hist.history["accuracy"][i], i)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

"""
Reference:
https://www.tensorflow.org/tutorials/distribute/parameter_server_training
"""
