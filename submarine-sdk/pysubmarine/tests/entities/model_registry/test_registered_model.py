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

from submarine.entities.model_registry import RegisteredModel, RegisteredModelTag


class TestRegisteredModel:
    default_data = {
        "name": "test",
        "creation_time": datetime.now(),
        "last_updated_time": datetime.now(),
        "description": "registered model description",
        "tags": [],
    }

    def _check(self, registered_model, name, creation_time, last_updated_time, description, tags):
        isinstance(registered_model, RegisteredModel)
        assert registered_model.name == name
        assert registered_model.creation_time == creation_time
        assert registered_model.last_updated_time == last_updated_time
        assert registered_model.description == description
        assert registered_model.tags == tags

    def test_creation_and_hydration(self):
        rm = RegisteredModel(
            self.default_data["name"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["description"],
            self.default_data["tags"],
        )
        self._check(
            rm,
            self.default_data["name"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["description"],
            self.default_data["tags"],
        )

    def test_with_tags(self):
        tag1 = RegisteredModelTag("tag1")
        tag2 = RegisteredModelTag("tag2")
        tags = [tag1, tag2]
        rm = RegisteredModel(
            self.default_data["name"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["description"],
            tags,
        )
        self._check(
            rm,
            self.default_data["name"],
            self.default_data["creation_time"],
            self.default_data["last_updated_time"],
            self.default_data["description"],
            [t.tag for t in tags],
        )
