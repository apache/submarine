# Copyright 2020 The Kubeflow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The code is mainly referenced from:
https://docs.seldon.io/projects/seldon-core/en/latest/examples/tfserving_mnist.html
And the parameters of the predictions call have been modified.
"""

import numpy as np
import requests
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from packaging.version import Version


def show_image(image):
    """
    show the image
    """
    two_d = (np.reshape(image, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


def rest_request_ambassador(endpoint="localhost:32080", prefix="/", arr=None):
    """
    request ambassador with rest
    """
    from tensorflow.python.ops.numpy_ops import np_config

    np_config.enable_numpy_behavior()
    payload = {
        "data": {
            "names": ["image_predictions"],
            "tensor": {"shape": [-1, 28, 28, 1], "values": arr.tolist()},
        }
    }
    response = requests.post(
        # you can also find a swagger ui in http://endpoint/prefix/api/v0.1/doc/
        "http://" + endpoint + prefix + "api/v0.1/predictions",
        json=payload,
    )
    print(response.text)
    # get the prediction
    print(f'TThe prediction is {np.argmax(response.json()["data"]["tensor"]["values"])}.')


# download datasets
if Version(tfds.__version__) > Version("3.1.0"):
    tfds.core.utils.gcs_utils._is_gcs_disabled = True
datasets, _ = tfds.load(name="mnist", with_info=True, as_supervised=True)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


# get first dataset
first_dataset = datasets["train"].map(scale).take(1)
for image, label in first_dataset:
    show_image(image)
    rest_request_ambassador(
        endpoint="localhost:32080",
        # This prefix you can find in VirtualService in istio, command like:
        # kubectl describe VirtualService -n submarine-user-test -l model-name=${model_name}
        prefix="/seldon/submarine-user-test/1/1/",
        arr=image,
    )
