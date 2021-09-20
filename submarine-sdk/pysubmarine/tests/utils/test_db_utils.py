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

import os

import mock

from submarine.store import DEFAULT_SUBMARINE_JDBC_URL
from submarine.utils import get_db_uri, set_db_uri
from submarine.utils.db_utils import _DB_URI_ENV_VAR, is_db_uri_set


def test_is_db_uri_set():
    env = {
        _DB_URI_ENV_VAR: DEFAULT_SUBMARINE_JDBC_URL,
    }
    with mock.patch.dict(os.environ, env):
        assert is_db_uri_set() is True


def test_set_db_uri():
    test_db_uri = "mysql+pymysql://submarine:password@localhost:3306/submarine_test"
    set_db_uri(test_db_uri)
    assert get_db_uri() == test_db_uri
    set_db_uri(None)


def test_get_db_uri():
    env = {
        _DB_URI_ENV_VAR: DEFAULT_SUBMARINE_JDBC_URL,
    }
    with mock.patch.dict(os.environ, env):
        assert get_db_uri() == DEFAULT_SUBMARINE_JDBC_URL
