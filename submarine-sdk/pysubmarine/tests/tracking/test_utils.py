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
from submarine.store.tracking.sqlalchemy_store import SqlAlchemyStore
from submarine.tracking.utils import (
    _JOB_ID_ENV_VAR,
    _TRACKING_URI_ENV_VAR,
    get_job_id,
    get_sqlalchemy_store,
)


def test_get_job_id():
    env = {
        _JOB_ID_ENV_VAR: "application_12346789",
    }
    with mock.patch.dict(os.environ, env):
        assert get_job_id() == "application_12346789"


def test_get_sqlalchemy_store():
    patch_create_engine = mock.patch("sqlalchemy.create_engine")
    uri = DEFAULT_SUBMARINE_JDBC_URL
    env = {_TRACKING_URI_ENV_VAR: uri}
    with mock.patch.dict(os.environ, env), patch_create_engine as mock_create_engine, mock.patch(
        "submarine.store.tracking.sqlalchemy_store.SqlAlchemyStore._initialize_tables"
    ):
        store = get_sqlalchemy_store(uri)
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri
    mock_create_engine.assert_called_once_with(uri, pool_pre_ping=True)
