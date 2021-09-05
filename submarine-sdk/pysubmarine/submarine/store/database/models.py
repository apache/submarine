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

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import (Integer, BigInteger, Boolean, Text, Column, PrimaryKeyConstraint,
                        String, ForeignKey, ForeignKeyConstraint)
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from submarine.entities import Metric, Param, Experiment
from submarine.entities.model_registry import (RegisteredModel, RegisteredModelTag,
                                               ModelVersion, ModelTag)
from submarine.entities.model_registry.model_version_stages import STAGE_NONE
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
    __tablename__ = "registered_model"

    name = Column(String(256), unique=True, nullable=False)
    """
    Name for registered models: Part of *Primary Key* for ``registered_model`` table.
    """

    creation_time = Column(DATETIME(fsp=3), default=datetime.now())
    """
    Creation time of registered models: default current time in milliseconds
    """

    last_updated_time = Column(DATETIME(fsp=3), nullable=True, default=None)
    """
    Last updated time of registered model
    """

    description = Column(String(5000), nullable=True, default="")
    """
    Description for registered model
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
                               description=self.description,
                               tags=[tag.to_submarine_entity for tag in self.registered_model_tag])


# +---------------------+-------+
# | name                | tag   |
# +---------------------+-------+
# | image_classfication | image |
# | image_classfication | major |
# | speech_recoginition | audio |
# +---------------------+-------+


class SqlRegisteredModelTag(Base):
    __tablename__ = 'registered_model_tag'

    name = Column(String(256),
                  ForeignKey("registered_model.name", onupdate="cascade", ondelete="cascade"))
    """
    Name for registered models: Part of *Primary Key* for ``registered_model_tag`` table. Refer to
    name of ``registered_model`` table.
    """

    tag = Column(String(256), nullable=False)
    """
    Registered model tag: `String` (limit 256 characters). Part of *Primary Key* for
    ``registered_model_tag`` table.
    """

    # linked entities
    registered_model = relationship(
        "SqlRegisteredModel", backref=backref("registered_model_tag", cascade="all")
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
    __tablename__ = 'model_version'

    name = Column(String(256), ForeignKey("registered_model.name", onupdate="cascade"))
    """
    Name for registered models: Part of *Primary Key* for ``registered_model_tag`` table. Refer to
    name of ``registered_model`` table.
    """

    version = Column(Integer, nullable=False)
    """
    Model version: Part of *Primary Key* for ``registered_model_tag`` table.
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

    creation_time = Column(DATETIME(fsp=3), default=datetime.now())
    """
    Creation time of this model version: default current time in milliseconds
    """

    last_updated_time = Column(DATETIME(fsp=3), nullable=True, default=None)
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
        "SqlRegisteredModel", backref=backref("model_version", cascade="all")
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
                            description=self.description,
                            tags=[tag.to_submarine_entity for tag in self.model_tag])


# +---------------------+---------+-----------------+
# | name                | version | tag             |
# +---------------------+---------+-----------------+
# | image_classfication | 1       | best            |
# | image_classfication | 1       | anomaly_support |
# | image_classfication | 2       | testing         |
# | speech_recoginition | 1       | best            |
# +---------------------+---------+-----------------+


class SqlModelTag(Base):
    __tablename__ = 'model_tag'

    name = Column(String(256), nullable=False)
    """
    Name for registered models: Part of *Foreign Key* for ``model_tag`` table. Refer to
    name of ``model_version`` table.
    """

    version = Column(Integer, nullable=False)
    """
    version of model: Part of *Foreign Key* for ``model_tag`` table. Refer to
    version of ``model_version`` table.
    """

    tag = Column(String(256), nullable=False)
    """
    tag of model version: `String` (limit 256 characters). Part of *Primary Key* for
    ``model_tag`` table.
    """

    # linked entities
    model_version = relationship(
        "SqlModelVersion",
        foreign_keys=[name, version],
        backref=backref("model_tag", cascade="all")
    )

    __table_args__ = (
        PrimaryKeyConstraint("name", "tag", name="model_tag_pk"),
        ForeignKeyConstraint(
            ("name", "version"),
            ("model_version.name", "model_version.version"),
            onupdate="cascade",
            ondelete="cascade",
        ),
    )

    def __repr__(self):
        return "<SqlRegisteredModelTag ({}, {}, {})>".format(self.name, self.version, self.tag)

    # entity mappers
    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.ModelTag`.
        """
        return ModelTag(self.tag)

# +--------------------+-----------------+-----------+---------------+-----------+-------------+
# | id                 | experiment_spec | create_by | create_time   | update_by | update_time |
# +--------------------+-----------------+-----------+---------------+-----------+-------------+
# | application_123456 | ...             | root      | 1595414873500 |           |             |
# +--------------------+-----------------+-----------+---------------+-----------+-------------+


class SqlExperiment(Base):
    __tablename__ = 'experiment'

    id = Column(String(64))
    """
    ID to which this metric belongs to: Part of *Primary Key* for ``experiment`` table.
    """
    experiment_spec = Column(Text)
    """
    The spec to create this experiment
    """
    create_by = Column(String(32))
    """
    This experiment is created by whom.
    """
    create_time = Column(DATETIME(fsp=3), default=datetime.now())
    """
    Datetime of this experiment be created
    """
    update_by = Column(String(32))
    """
    This experiment is created by whom.
    """
    update_time = Column(DATETIME(fsp=3))
    """
    Datetime of this experiment be updated
    """

    __table_args__ = (PrimaryKeyConstraint('id'),)

    def __repr__(self):
        return '<SqlMetric({}, {}, {}, {}, {}, {})>'.format(self.id,
                                                            self.experiment_spec,
                                                            self.create_by,
                                                            self.create_time,
                                                            self.update_by,
                                                            self.update_time)

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.Experiment`.
        """
        return Experiment(id=self.id,
                          experiment_spec=self.experiment_spec,
                          create_by=self.create_by,
                          create_time=self.create_time,
                          update_by=self.update_by,
                          update_time=self.update_time)

# +--------------------+-------+-------------------+--------------+---------------+------+--------+
# | id                 | key   | value             | worker_index | timestamp     | step | is_nan |
# +--------------------+-------+-------------------+--------------+---------------+------+--------+
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595414873838 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595472286360 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595414632967 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595415075067 |    0 |      0 |
# +--------------------+-------+-------------------+--------------+---------------+------+--------+


class SqlMetric(Base):
    __tablename__ = 'metric'

    id = Column(String(64), ForeignKey("experiment.id", onupdate="cascade", ondelete="cascade"))
    """
    ID to which this metric belongs to: *Foreign Key* for ``experiment`` table.
    Part of *Primary Key* for ``metric`` table.
    """
    key = Column(String(190))
    """
    Metric key: `String` (limit 190 characters). Part of *Primary Key* for ``metric`` table.
    """
    value = Column(sa.types.Float(precision=53), nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    worker_index = Column(String(64))
    """
    Metric worker_index: `String` (limit 32 characters). Part of *Primary Key* for
    ``metric`` table.
    """
    timestamp = Column(DATETIME(fsp=3), default=datetime.now())
    """
    Timestamp recorded for this metric entry: `DATETIME`. Part of *Primary Key* for
    ``metric`` table.
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
    __tablename__ = 'param'

    id = Column(String(64), ForeignKey("experiment.id", onupdate="cascade", ondelete="cascade"))
    """
    ID to which this parameter belongs to: *Foreign Key* for ``experiment`` table.
    Part of *Primary Key* for ``param`` table.
    """
    key = Column(String(190))
    """
    Param key: `String` (limit 190 characters). Part of *Primary Key* for ``param`` table.
    """
    value = Column(String(190), nullable=False)
    """
    Param value: `String` (limit 190 characters). Defined as *Non-null* in schema.
    """
    worker_index = Column(String(32), nullable=False)
    """
    Param worker_index: `String` (limit 32 characters). Part of *Primary Key* for
    ``metric`` table.
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
