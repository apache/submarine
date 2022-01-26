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

from datetime import datetime

from submarine.entities.model_registry import ModelVersion, ModelVersionTag
from submarine.entities.model_registry.model_stages import STAGE_NONE


class TestModelVersion:
    default_data = {
        "name": "test",
        "version": 1,
        "id": "1f94b4fadbe144ea8ced0ce195855cfc",
        "user_id": "admin",
        "experiment_id": "experiment_1",
        "model_type": "tensorflow",
        "current_stage": STAGE_NONE,
        "creation_time": datetime.now(),
        "last_updated_time": datetime.now(),
        "dataset": "test",
        "description": "registered model description",
        "tags": [],
    }

    def _check(
        self,
        model_metadata,
        name,
        version,
        id,
        user_id,
        experiment_id,
        model_type,
        current_stage,
        creation_time,
        last_updated_time,
        dataset,
        description,
        tags,
    ):
        isinstance(model_metadata, ModelVersion)
        assert model_metadata.name == name
        assert model_metadata.version == version
        assert model_metadata.id == id
        assert model_metadata.user_id == user_id
        assert model_metadata.experiment_id == experiment_id
        assert model_metadata.model_type == model_type
        assert model_metadata.current_stage == current_stage
        assert model_metadata.creation_time == creation_time
        assert model_metadata.last_updated_time == last_updated_time
        assert model_metadata.dataset == dataset
        assert model_metadata.description == description
        assert model_metadata.tags == tags

    def test_creation_and_hydration(self):
        mv = ModelVersion(
            self.default_data["name"],
            self.default_data["version"],
            self.default_data["id"],
            self.default_data["user_id"],
            self.default_data["experiment_id"],
            self.default_data["model_type"],
            self.default_data["current_stage"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["dataset"],
            self.default_data["description"],
            self.default_data["tags"],
        )
        self._check(
            mv,
            self.default_data["name"],
            self.default_data["version"],
            self.default_data["id"],
            self.default_data["user_id"],
            self.default_data["experiment_id"],
            self.default_data["model_type"],
            self.default_data["current_stage"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["dataset"],
            self.default_data["description"],
            self.default_data["tags"],
        )

    def test_with_tags(self):
        tag1 = ModelVersionTag("tag1")
        tag2 = ModelVersionTag("tag2")
        tags = [tag1, tag2]
        mv = ModelVersion(
            self.default_data["name"],
            self.default_data["version"],
            self.default_data["id"],
            self.default_data["user_id"],
            self.default_data["experiment_id"],
            self.default_data["model_type"],
            self.default_data["current_stage"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["dataset"],
            self.default_data["description"],
            tags,
        )
        self._check(
            mv,
            self.default_data["name"],
            self.default_data["version"],
            self.default_data["id"],
            self.default_data["user_id"],
            self.default_data["experiment_id"],
            self.default_data["model_type"],
            self.default_data["current_stage"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["dataset"],
            self.default_data["description"],
            [t.tag for t in tags],
        )
