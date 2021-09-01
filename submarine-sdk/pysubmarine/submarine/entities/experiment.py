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


class Experiment(_SubmarineObject):
    """
    Experiment object.
    """

    def __init__(self, id, experiment_spec, create_by, create_time, update_by,
                 update_time):
        self._id = id
        self._experiment_spec = experiment_spec
        self._create_by = create_by
        self._create_time = create_time
        self._update_by = update_by
        self._update_time = update_time

    @property
    def id(self):
        """String ID of the experiment."""
        return self._id

    @property
    def experiment_spec(self):
        """String of the experiment spec."""
        return self._experiment_spec

    @property
    def create_by(self):
        """String name of created user id."""
        return self.create_by

    @property
    def create_time(self):
        """Integer of create time."""
        return self._create_time

    @property
    def update_by(self):
        """String name of updated user id"."""
        return self._update_by

    @property
    def update_time(self):
        """Integer of update time."""
        return self._update_time
