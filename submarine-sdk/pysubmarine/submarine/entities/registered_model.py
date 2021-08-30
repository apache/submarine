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

from submarine.entities._submarine_object import _SubmarineObject


class RegisteredModel(_SubmarineObject):
    """
    Registered Model object.
    """

    def __init__(self,
                 name,
                 creation_time,
                 last_updated_time,
                 description=None):
        super().__init__()
        self._name = name
        self._creation_time = creation_time
        self._last_updated_time = last_updated_time
        self._description = description

    @property
    def name(self):
        """String. Registered model name."""
        return self._name

    @property
    def creation_time(self):
        """Integer. Model version creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_time(self):
        """Integer. Timestamp of last update for this model version (milliseconds since the Unix
        epoch)."""
        return self._last_updated_time

    @property
    def description(self):
        """String. Description"""
        return self._description
