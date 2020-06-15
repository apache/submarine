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

import pprint


class _SubmarineObject:

    def __iter__(self):
        # Iterate through list of properties and yield as key -> value
        for prop in self._properties():
            yield prop, self.__getattribute__(prop)

    @classmethod
    def _properties(cls):
        return sorted(
            [p for p in cls.__dict__ if isinstance(getattr(cls, p), property)])

    @classmethod
    def from_dictionary(cls, the_dict):
        filtered_dict = {
            key: value
            for key, value in the_dict.items()
            if key in cls._properties()
        }
        return cls(**filtered_dict)

    def __repr__(self):
        return to_string(self)


def to_string(obj):
    return _SubmarineObjectPrinter().to_string(obj)


def get_classname(obj):
    return type(obj).__name__


class _SubmarineObjectPrinter:

    def __init__(self):
        super(_SubmarineObjectPrinter, self).__init__()
        self.printer = pprint.PrettyPrinter()

    def to_string(self, obj):
        if isinstance(obj, _SubmarineObject):
            return "<%s: %s>" % (get_classname(obj),
                                 self._entity_to_string(obj))
        return self.printer.pformat(obj)

    def _entity_to_string(self, entity):
        return ", ".join(
            ["%s=%s" % (key, self.to_string(value)) for key, value in entity])
