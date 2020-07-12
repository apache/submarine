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

import pytest
from submarine.exceptions import SubmarineException
from submarine.utils.validation import (_validate_db_type_string,
                                        _validate_length_limit,
                                        _validate_metric_name,
                                        _validate_param_name)

GOOD_METRIC_OR_PARAM_NAMES = [
    "a",
    "Ab-5_",
    "a/b/c",
    "a.b.c",
    ".a",
    "b.",
    "a..a/._./o_O/.e.",
    "a b/c d",
]
BAD_METRIC_OR_PARAM_NAMES = [
    "",
    ".",
    "/",
    "..",
    "//",
    "a//b",
    "a/./b",
    "/a",
    "a/",
    ":",
    "\\",
    "./",
    "/./",
]


def test_validate_metric_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_metric_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(SubmarineException, match="Invalid metric name"):
            _validate_metric_name(bad_name)


def test_validate_param_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_param_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(SubmarineException, match="Invalid parameter name"):
            _validate_param_name(bad_name)


def test__validate_length_limit():
    limit = 10
    key = "key"
    good_value = "test-12345"
    bad_value = "test-123456789"
    _validate_length_limit(key, limit, good_value)
    with pytest.raises(SubmarineException, match="which exceeded length limit"):
        _validate_length_limit(key, limit, bad_value)


def test_db_type():
    for db_type in ["mysql", "mssql", "postgresql", "sqlite"]:
        # should not raise an exception
        _validate_db_type_string(db_type)

    # error cases
    for db_type in ["MySQL", "mongo", "cassandra", "sql", ""]:
        with pytest.raises(SubmarineException) as e:
            _validate_db_type_string(db_type)
        assert "Invalid database engine" in e.value.message
