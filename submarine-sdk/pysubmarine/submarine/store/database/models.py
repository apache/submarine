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

import time
from typing import Any

import sqlalchemy as sa
from sqlalchemy import (Integer, BigInteger, Boolean, Column, PrimaryKeyConstraint,
                        String, ForeignKey, ForeignKeyConstraint)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from submarine.entities import (Metric, Param, RegisteredModel, RegisteredModelTag,
                                ModelVersion, ModelVersionTag)
from submarine.entities.model_version_stages import STAGE_NONE
Base = declarative_base()

# +---------------------+---------------+-------------------+-------------+
# | name                | creation_time | last_updated_time | description |
# +---------------------+---------------+-------------------+-------------+
# | image_classfication | 1595414873838 | 1595414873838     | ...         |
# | speech_recoginition | 1595472153245 | 1595472286360     | ...         |
# +---------------------+---------------+-------------------+-------------+

# Base class in sqlalchemy is a dynamic type
Base: Any = declarative_base()

class SqlRegisteredModel(Base):
    __tablename__ = "registered_models"

    name = Column(String(256), unique=True, nullable=False)
    """
    Name for registered models: Part of *Primary Key* for ``registered_models`` table.
    """

    creation_time = Column(BigInteger, default=lambda: int(time.time() * 1000))
    """
    Creation time of registered models: default current time in milliseconds
    """

    last_updated_time = Column(BigInteger, nullable=True, default=None)
    """
    Last updated time of registered models
    """

    description = Column(String(5000), nullable=True, default="")
    """
    Description for registered models
    """

    __table_args__ = (PrimaryKeyConstraint("name", name="registered_model_pk"),)

    def __repr__(self):
        return "<SqlRegisteredModel ({}, {}, {}, {})>".format(
            self.name, self.creation_time, self.last_updated_time, self.description
        )

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.RegisteredModel`.
        """
        return RegisteredModel(name=self.name,
                               creation_time=self.creation_time,
                               last_updated_time=self.last_updated_time,
                               description=self.description)


# +---------------------+-------+
# | name                | tag   |
# +---------------------+-------+
# | image_classfication | image |
# | image_classfication | major |
# | speech_recoginition | audio |
# +---------------------+-------+


class SqlRegisteredModelTag(Base):
    __tablename__ = 'registered_model_tags'

    name = Column(String(256), ForeignKey("registered_models.name", onupdate="cascade"))
    """
    Name for registered models: Part of *Primary Key* for ``registered_model_tags`` table. Refer to
    name of ``registered_models`` table.
    """

    tag = Column(String(256), nullable=False)
    """
    Registered model tag: `String` (limit 256 characters). Part of *Primary Key* for
    ``registered_model_tags`` table.
    """

    # linked entities
    registered_model = relationship(
        "SqlRegisteredModel", backref=backref("registered_model_tags", cascade="all")
    )

    __table_args__ = (PrimaryKeyConstraint("name", "tag", name="registered_model_tag_pk"),)

    def __repr__(self):
        return "<SqlRegisteredModelTag ({}, {})>".format(self.name, self.tag)

    # entity mappers
    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.RegisteredModelTag`.
        """
        return RegisteredModelTag(self.tag)

# +---------------------+---------+-----+-------------------------------+-----+
# | name                | version | ... | source                        | ... |
# +---------------------+---------+-----+-------------------------------+-----+
# | image_classfication | 1       | ... | s3://submarine/ResNet50/1/    | ... |
# | image_classfication | 2       | ... | s3://submarine/DenseNet121/2/ | ... |
# | speech_recoginition | 1       | ... | s3://submarine/ASR/1/         | ... |
# +---------------------+---------+-----+-------------------------------+-----+


class SqlModelVersion(Base):
    __tablename__ = 'model_versions'

    name = Column(String(256), ForeignKey("registered_models.name", onupdate="cascade"))
    """
    Name for registered models: Part of *Primary Key* for ``registered_model_tags`` table. Refer to
    name of ``registered_models`` table.
    """

    version = Column(Integer, nullable=False)
    """
    Model version: Part of *Primary Key* for ``registered_model_tags`` table.
    """

    user_id = Column(String(64), nullable=False)
    """
    ID to whom this model is created
    """

    experiment_id = Column(String(64), nullable=False)
    """
    ID to which this model belongs to
    """

    current_stage = Column(String(20), default=STAGE_NONE)
    """
    Current stage of this model: it can be `None`, `Staging`, `Production` and `Achieved`
    """

    creation_time = Column(BigInteger, default=lambda: int(time.time() * 1000))
    """
    Creation time of this model version: default current time in milliseconds
    """

    last_updated_time = Column(BigInteger, nullable=True, default=None)
    """
    Last updated time of this model version
    """

    source = Column(String(512), nullable=True, default=None)
    """
    Source of model: database link refer to this model
    """

    dataset = Column(String(256), nullable=True, default=None)
    """
    Dataset used for this model.
    """

    description = Column(String(5000), nullable=True)
    """
    Description for model version.
    """

    # linked entities
    registered_model = relationship(
        "SqlRegisteredModel", backref=backref("model_versions", cascade="all")
    )

    __table_args__ = (PrimaryKeyConstraint("name", "version", name="model_version_pk"),)

    def __repr__(self):
        return "<SqlModelVersion ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})>".format(
            self.name, self.version, self.user_id, self.experiment_id, self.current_stage,
            self.creation_time, self.last_updated_time, self.source, self.dataset, self.description
        )

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.RegisteredModel`.
        """
        return ModelVersion(name=self.name,
                            version=self.version,
                            user_id=self.user_id,
                            experiment_id=self.experiment_id,
                            current_stage=self.current_stage,
                            creation_time=self.creation_time,
                            last_updated_time=self.last_updated_time,
                            source=self.source,
                            dataset=self.dataset,
                            description=self.description)


# +---------------------+---------+-----------------+
# | name                | version | tag             |
# +---------------------+---------+-----------------+
# | image_classfication | 1       | best            |
# | image_classfication | 1       | anomaly_support |
# | image_classfication | 2       | testing         |
# | speech_recoginition | 1       | best            |
# +---------------------+---------+-----------------+


class SqlModelVersionTag(Base):
    __tablename__ = 'model_version_tags'

    name = Column(String(256), nullable=False)
    """
    Name for registered models: Part of *Foreign Key* for ``model_version_tags`` table. Refer to
    name of ``model_versions`` table.
    """

    version = Column(Integer, nullable=False)
    """
    version of model: Part of *Foreign Key* for ``model_version_tags`` table. Refer to
    version of ``model_versions`` table.
    """

    tag = Column(String(256), nullable=False)
    """
    tag of model version: `String` (limit 256 characters). Part of *Primary Key* for
    ``model_version_tags`` table.
    """

    # linked entities
    model_version = relationship(
        "SqlModelVersion",
        foreign_keys=[name, version],
        backref=backref("model_version_tags", cascade="all")
    )

    __table_args__ = (
        PrimaryKeyConstraint("name", "tag", name="model_version_tag_pk"),
        ForeignKeyConstraint(
            ("name", "version"),
            ("model_versions.name", "model_versions.version"),
            onupdate="cascade",
        ),
    )

    def __repr__(self):
        return "<SqlRegisteredModelTag ({}, {}, {})>".format(self.name, self.version, self.tag)

    # entity mappers
    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.ModelVersionTag`.
        """
        return ModelVersionTag(self.tag)

# +--------------------+-------+-------------------+--------------+---------------+------+--------+
# | id                 | key   | value             | worker_index | timestamp     | step | is_nan |
# +--------------------+-------+-------------------+--------------+---------------+------+--------+
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595414873838 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595472286360 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595414632967 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595415075067 |    0 |      0 |
# +--------------------+-------+-------------------+--------------+---------------+------+--------+


class SqlMetric(Base):
    __tablename__ = "metrics"

    id = Column(String(64))
    """
    ID to which this metric belongs to: Part of *Primary Key* for ``metrics`` table.
    """
    key = Column(String(190))
    """
    Metric key: `String` (limit 190 characters). Part of *Primary Key* for ``metrics`` table.
    """
    value = Column(sa.types.Float(precision=53), nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    worker_index = Column(String(64))
    """
    Metric worker_index: `String` (limit 32 characters). Part of *Primary Key* for
    ``metrics`` table.
    """
    timestamp = Column(BigInteger, default=lambda: int(time.time()))
    """
    Timestamp recorded for this metric entry: `BigInteger`. Part of *Primary Key* for
    ``metrics`` table.
    """
    step = Column(BigInteger, default=0, nullable=False)
    """
    Step recorded for this metric entry: `BigInteger`.
    """
    is_nan = Column(Boolean, nullable=False, default=False)
    """
    True if the value is in fact NaN.
    """

    __table_args__ = (
        PrimaryKeyConstraint("id", "key", "timestamp", "worker_index", name="metric_pk"),
    )

    def __repr__(self):
        return "<SqlMetric({}, {}, {}, {}, {})>".format(
            self.key, self.value, self.worker_index, self.timestamp, self.step
        )

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.Metric`.
        """
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            worker_index=self.worker_index,
            timestamp=self.timestamp,
            step=self.step,
        )


# +-----------------------+----------+-------+--------------+
# | id                    | key      | value | worker_index |
# +-----------------------+----------+-------+--------------+
# | application_123456898 | max_iter | 100   | worker-1     |
# | application_123456898 | alpha    | 10    | worker-1     |
# | application_123456898 | n_jobs   | 5     | worker-1     |
# +-----------------------+----------+-------+--------------+


class SqlParam(Base):
    __tablename__ = "params"

    id = Column(String(64))
    """
    ID to which this parameter belongs to: Part of *Primary Key* for ``params`` table.
    """
    key = Column(String(190))
    """
    Param key: `String` (limit 190 characters). Part of *Primary Key* for ``params`` table.
    """
    value = Column(String(190), nullable=False)
    """
    Param value: `String` (limit 190 characters). Defined as *Non-null* in schema.
    """
    worker_index = Column(String(32), nullable=False)
    """
    Param worker_index: `String` (limit 32 characters). Part of *Primary Key* for
    ``metrics`` table.
    """

    __table_args__ = (PrimaryKeyConstraint("id", "key", "worker_index", name="param_pk"),)

    def __repr__(self):
        return "<SqlParam({}, {}, {})>".format(self.key, self.value, self.worker_index)

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.Param`.
        """
        return Param(key=self.key, value=self.value, worker_index=self.worker_index)
