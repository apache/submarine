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

import unittest
from os import environ

import pytest

import submarine
from submarine.store.database import models
from submarine.store.database.models import SqlMetric, SqlParam
from submarine.tracking import utils

JOB_ID = "application_123456789"


@pytest.mark.e2e
class TestTracking(unittest.TestCase):
    def setUp(self):
        environ["JOB_ID"] = JOB_ID
        submarine.set_tracking_uri(
            "mysql+pymysql://submarine_test:password_test@localhost:3306/submarine_test"
        )
        self.tracking_uri = utils.get_tracking_uri()
        self.store = utils.get_sqlalchemy_store(self.tracking_uri)

    def tearDown(self):
        submarine.set_tracking_uri(None)
        models.Base.metadata.drop_all(self.store.engine)

    def log_param(self):
        submarine.log_param("name_1", "a")
        # Validate params
        with self.store.ManagedSessionMaker() as session:
            params = session.query(SqlParam).options().filter(SqlParam.id == JOB_ID).all()
            assert params[0].key == "name_1"
            assert params[0].value == "a"
            assert params[0].id == JOB_ID

    def test_log_metric(self):
        submarine.log_metric("name_1", 5)
        submarine.log_metric("name_1", 6)
        # Validate params
        with self.store.ManagedSessionMaker() as session:
            metrics = session.query(SqlMetric).options().filter(SqlMetric.id == JOB_ID).all()
            assert len(metrics) == 2
            assert metrics[0].key == "name_1"
            assert metrics[0].value == 5
            assert metrics[0].id == JOB_ID
            assert metrics[1].value == 6
