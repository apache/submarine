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
from submarine.ml.pytorch.metric import get_metric_fn


def test_get_metric_fn():
    metric_keys = ['f1_score', 'accuracy', 'roc_auc', 'precision', 'recall']
    invalid_metric_keys = ['NotExistMetric']

    for key in metric_keys:
        get_metric_fn(key)

    for key_invalid in invalid_metric_keys:
        with pytest.raises(ValueError, match='Invalid metric_key:'):
            get_metric_fn(key_invalid)
