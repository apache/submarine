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

import pytest

from submarine.ml.pytorch.optimizer import get_optimizer


def test_get_optimizer():
    optimizer_keys = ["adam", "adagrad", "sgd"]
    invalid_optimizer_keys = ["adddam"]

    for key in optimizer_keys:
        get_optimizer(key)

    for key_invalid in invalid_optimizer_keys:
        with pytest.raises(ValueError, match="Invalid optimizer_key:"):
            get_optimizer(key_invalid)
