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
from datetime import datetime

import pytest

import submarine
from submarine.entities import Metric, Param
from submarine.store.database import models
from submarine.store.database.models import SqlExperiment, SqlMetric, SqlParam
from submarine.store.sqlalchemy_store import SqlAlchemyStore

JOB_ID = "application_123456789"


@pytest.mark.e2e
class TestSqlAlchemyStore(unittest.TestCase):
    def setUp(self):
        submarine.set_db_uri(
            "mysql+pymysql://submarine_test:password_test@localhost:3306/submarine_test"
        )
        self.db_uri = submarine.get_db_uri()
        self.store = SqlAlchemyStore(self.db_uri)
        # TODO(KUAN-HSUN-LI): use submarine.tracking.fluent to support experiment create
        with self.store.ManagedSessionMaker() as session:
            instance = SqlExperiment(
                id=JOB_ID,
                experiment_spec='{"value": 1}',
                create_by="test",
                create_time=datetime.now(),
                update_by=None,
                update_time=None,
            )
            session.add(instance)
            session.commit()

    def tearDown(self):
        submarine.set_db_uri(None)
        models.Base.metadata.drop_all(self.store.engine)

    def test_log_param(self):
        param1 = Param("name_1", "a", "worker-1")
        self.store.log_param(JOB_ID, param1)

        # Validate params
        with self.store.ManagedSessionMaker() as session:
            params = session.query(SqlParam).options().filter(SqlParam.id == JOB_ID).all()
            assert params[0].key == "name_1"
            assert params[0].value == "a"
            assert params[0].worker_index == "worker-1"
            assert params[0].id == JOB_ID

    def test_log_metric(self):
        metric1 = Metric("name_1", 5, "worker-1", datetime.now(), 0)
        metric2 = Metric("name_1", 6, "worker-2", datetime.now(), 0)
        self.store.log_metric(JOB_ID, metric1)
        self.store.log_metric(JOB_ID, metric2)

        # Validate metrics
        with self.store.ManagedSessionMaker() as session:
            metrics = session.query(SqlMetric).options().filter(SqlMetric.id == JOB_ID).all()
            assert len(metrics) == 2
            assert metrics[0].key == "name_1"
            assert metrics[0].value == 5
            assert metrics[0].worker_index == "worker-1"
            assert metrics[0].id == JOB_ID
            assert metrics[1].value == 6
            assert metrics[1].worker_index == "worker-2"
