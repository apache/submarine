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

from submarine.exceptions import SubmarineException

STAGE_NONE = "None"
STAGE_DEVELOPING = "Developing"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"

STAGE_DELETED_INTERNAL = "Deleted_Internal"

ALL_STAGES = [STAGE_NONE, STAGE_DEVELOPING, STAGE_PRODUCTION, STAGE_ARCHIVED]
_CANONICAL_MAPPING = {stage.lower(): stage for stage in ALL_STAGES}


def get_canonical_stage(stage):
    key = stage.lower()
    if key not in _CANONICAL_MAPPING:
        raise SubmarineException(f"Invalid Model Version stage {stage}.")
    return _CANONICAL_MAPPING[key]
