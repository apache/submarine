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

import submarine.tracking.fluent
import submarine.utils as utils
from submarine.client.api.experiment_client import ExperimentClient
from submarine.models.client import ModelsClient

log_param = submarine.tracking.fluent.log_param
log_metric = submarine.tracking.fluent.log_metric
save_model = submarine.tracking.fluent.save_model
set_db_uri = utils.set_db_uri
get_db_uri = utils.get_db_uri

__all__ = [
    "log_metric",
    "log_param",
    "save_model",
    "set_db_uri",
    "get_db_uri",
    "ExperimentClient",
    "ModelsClient",
]
