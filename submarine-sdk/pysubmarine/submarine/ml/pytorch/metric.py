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

from sklearn import metrics


class MetricKey:
    F1_SCORE = 'f1_score'
    ACCURACY = 'accuracy'
    ROC_AUC = 'roc_auc'
    PRECISION = 'precision'
    RECALL = 'recall'


def get_metric_fn(key):
    key = key.lower()
    if key == MetricKey.F1_SCORE:
        return metrics.f1_score
    if key == MetricKey.ACCURACY:
        return metrics.accuracy_score
    if key == MetricKey.ROC_AUC:
        return metrics.roc_auc_score
    if key == MetricKey.PRECISION:
        return metrics.precision_score
    if key == MetricKey.RECALL:
        return metrics.recall_score
    raise ValueError('Invalid metric_key:', key)
