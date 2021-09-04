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

import time
from submarine.entities.model_registry.model_version import ModelVersion
from submarine.entities.model_registry.model_tag import ModelTag
from submarine.entities.model_registry.model_version_stages import STAGE_NONE


class TestModelVersion():
    default_data = {
        "name": "test",
        "version": 1,
        "user_id": "admin",
        "experiment_id": "experiment_1",
        "current_stage": STAGE_NONE,
        "creation_time": int(time.time()),
        "last_updated_time": int(time.time()),
        "source": "path/to/source",
        "dataset": "test",
        "description": "registered model description",
        "tags": []
    }

    def _check(
        self,
        model_version,
        name,
        version,
        user_id,
        experiment_id,
        current_stage,
        creation_time,
        last_updated_time,
        source,
        dataset,
        description,
        tags
    ):
        isinstance(model_version, ModelVersion)
        assert model_version.name == name
        assert model_version.version == version
        assert model_version.user_id == user_id
        assert model_version.experiment_id == experiment_id
        assert model_version.current_stage == current_stage
        assert model_version.creation_time == creation_time
        assert model_version.last_updated_time == last_updated_time
        assert model_version.source == source
        assert model_version.dataset == dataset
        assert model_version.description == description
        assert model_version.tags == tags

    def test_creation_and_hydration(self):
        mv = ModelVersion(self.default_data["name"],
                          self.default_data["version"],
                          self.default_data["user_id"],
                          self.default_data["experiment_id"],
                          self.default_data["current_stage"],
                          self.default_data["creation_time"],
                          self.default_data["last_updated_time"],
                          self.default_data["source"],
                          self.default_data["dataset"],
                          self.default_data["description"],
                          self.default_data["tags"])
        self._check(mv,
                    self.default_data["name"],
                    self.default_data["version"],
                    self.default_data["user_id"],
                    self.default_data["experiment_id"],
                    self.default_data["current_stage"],
                    self.default_data["creation_time"],
                    self.default_data["last_updated_time"],
                    self.default_data["source"],
                    self.default_data["dataset"],
                    self.default_data["description"],
                    self.default_data["tags"])

    def test_with_tags(self):
        tag1 = ModelVersionTag("tag1")
        tag2 = ModelVersionTag("tag2")
        tags = [tag1, tag2]
        mv = ModelVersion(self.default_data["name"],
                          self.default_data["version"],
                          self.default_data["user_id"],
                          self.default_data["experiment_id"],
                          self.default_data["current_stage"],
                          self.default_data["creation_time"],
                          self.default_data["last_updated_time"],
                          self.default_data["source"],
                          self.default_data["dataset"],
                          self.default_data["description"],
                          tags)
        self._check(mv,
                    self.default_data["name"],
                    self.default_data["version"],
                    self.default_data["user_id"],
                    self.default_data["experiment_id"],
                    self.default_data["current_stage"],
                    self.default_data["creation_time"],
                    self.default_data["last_updated_time"],
                    self.default_data["source"],
                    self.default_data["dataset"],
                    self.default_data["description"],
                    [t.tag for t in tags])
