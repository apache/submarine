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

import submarine
from os import environ
from submarine.store.database.models import SqlMetric, SqlParam
from submarine.tracking import utils
from submarine.store.database import models
from submarine.entities import Metric, Param

import time
import unittest

JOB_NAME = "application_123456789"


class TestSqlAlchemyStore(unittest.TestCase):
    def setUp(self):
        submarine.set_tracking_uri(
            "mysql+pymysql://submarine_test:password_test@localhost:3306/submarine_test")
        self.tracking_uri = utils.get_tracking_uri()
        self.store = utils.get_sqlalchemy_store(self.tracking_uri)

    def tearDown(self):
        submarine.set_tracking_uri(None)
        models.Base.metadata.drop_all(self.store.engine)

    def test_log_param(self):
        param1 = Param("name_1", "a", "worker-1")
        self.store.log_param(JOB_NAME, param1)

        # Validate params
        with self.store.ManagedSessionMaker() as session:
            params = session \
                .query(SqlParam) \
                .options() \
                .filter(SqlParam.job_name == JOB_NAME).all()
            assert params[0].key == "name_1"
            assert params[0].value == "a"
            assert params[0].worker_index == "worker-1"
            assert params[0].job_name == JOB_NAME

    def test_log_metric(self):
        metric1 = Metric("name_1", 5, "worker-1", int(time.time()), 0)
        metric2 = Metric("name_1", 6, "worker-2", int(time.time()), 0)
        self.store.log_metric(JOB_NAME, metric1)
        self.store.log_metric(JOB_NAME, metric2)

        # Validate params
        with self.store.ManagedSessionMaker() as session:
            metrics = session \
                .query(SqlMetric) \
                .options() \
                .filter(SqlMetric.job_name == JOB_NAME).all()
            assert len(metrics) == 2
            assert metrics[0].key == "name_1"
            assert metrics[0].value == 5
            assert metrics[0].worker_index == "worker-1"
            assert metrics[0].job_name == JOB_NAME
            assert metrics[1].value == 6
            assert metrics[1].worker_index == "worker-2"
