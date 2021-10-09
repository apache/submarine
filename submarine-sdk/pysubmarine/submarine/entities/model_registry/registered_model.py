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

from typing import Type

from sqlalchemy.sql.schema import Column

from submarine.entities._submarine_object import _SubmarineObject


class RegisteredModel(_SubmarineObject):
    """
    Registered model object.
    """

    def __init__(
        self,
        name: Type[Column],
        creation_time: Type[Column],
        last_updated_time: Type[Column],
        description=None,
        tags=None,
    ):
        self._name = name
        self._creation_time = creation_time
        self._last_updated_time = last_updated_time
        self._description = description
        self._tags = [tag.tag for tag in (tags or [])]

    @property
    def name(self):
        """String. Registered model name."""
        return self._name

    @property
    def creation_time(self):
        """Datetime object. Registered model creation datetime."""
        return self._creation_time

    @property
    def last_updated_time(self):
        """Datetime object. Datetime of last update for this model."""
        return self._last_updated_time

    @property
    def description(self):
        """String. Description"""
        return self._description

    @property
    def tags(self):
        """List of strings"""
        return self._tags
