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
from os import environ

import pytest
import tensorflow

import submarine
from submarine.artifacts.repository import Repository
from submarine.store.database import models
from submarine.store.database.models import SqlExperiment, SqlMetric, SqlParam
from submarine.tracking.client import SubmarineClient

from .tf_model import LinearNNModel

JOB_ID = "application_123456789"
REGISTERED_MODEL_NAME = "registerd_model_name"
MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"


@pytest.mark.e2e
class TestTracking(unittest.TestCase):
    def setUp(self):
        environ["JOB_ID"] = JOB_ID
        submarine.set_db_uri(
            "mysql+pymysql://submarine_test:password_test@localhost:3306/submarine_test"
        )
        self.db_uri = submarine.get_db_uri()
        self.client = SubmarineClient(
            db_uri=self.db_uri,
            s3_registry_uri=MLFLOW_S3_ENDPOINT_URL,
        )
        from submarine.store.tracking.sqlalchemy_store import SqlAlchemyStore

        self.store = SqlAlchemyStore(self.db_uri)
        from submarine.store.model_registry.sqlalchemy_store import SqlAlchemyStore

        self.model_registry = SqlAlchemyStore(self.db_uri)
        # TODO: use submarine.tracking.fluent to support experiment create
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
        environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        environ["AWS_ACCESS_KEY_ID"] = "submarine_minio"
        environ["AWS_SECRET_ACCESS_KEY"] = "submarine_minio"
        Repository().delete_folder(f"experiment/{JOB_ID}")
        Repository().delete_folder(f"registry/{REGISTERED_MODEL_NAME}")

    def test_log_param(self):
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

    @pytest.mark.skipif(tensorflow.version.VERSION < "2.0", reason="using tensorflow 2")
    def test_save_model(self):
        input_arr = tensorflow.random.uniform((1, 5))
        model = LinearNNModel()
        model(input_arr)
        self.client.save_model(model, "tensorflow", "name_1", REGISTERED_MODEL_NAME)
        self.client.save_model(model, "tensorflow", "name_2", REGISTERED_MODEL_NAME)
        # Validate model_versions
        model_versions = self.model_registry.list_model_versions(REGISTERED_MODEL_NAME)
        assert len(model_versions) == 2
        assert model_versions[0].name == REGISTERED_MODEL_NAME
        assert model_versions[0].version == 1
        assert model_versions[1].name == REGISTERED_MODEL_NAME
        assert model_versions[1].version == 2
