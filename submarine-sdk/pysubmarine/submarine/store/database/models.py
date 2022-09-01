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
from typing import List

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, relationship

from submarine.entities import Experiment, Metric, Param
from submarine.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelTag,
)
from submarine.entities.model_registry.model_stages import STAGE_NONE

# Base class in sqlalchemy is a dynamic type
Base = declarative_base()

# +----------+-------------------------+-------------------------+-------------+
# | name     | creation_time           | last_updated_time       | description |
# +----------+-------------------------+-------------------------+-------------+
# | ResNet50 | 2021-08-31 11:11:11.111 | 2021-09-02 11:11:11.111 | ...         |
# | BERT     | 2021-08-31 16:16:16.166 | 2021-08-31 20:20:20.200 | ...         |
# +----------+-------------------------+-------------------------+-------------+


class SqlRegisteredModel(Base):
    __tablename__ = "registered_model"

    name = Column(String(256), unique=True)
    """
    Name of registered model: Part of *Primary Key* for ``registered_model`` table.
    """

    creation_time = Column(DATETIME(fsp=3), default=datetime.now())
    """
    Creation time of registered model: default current time in milliseconds.
    """

    last_updated_time = Column(DATETIME(fsp=3), nullable=True)
    """
    Last updated time of registered model.
    """

    description = Column(String(5000), nullable=True, default=None)
    """
    Description for registered model.
    """

    tags: Mapped[List["SqlRegisteredModelTag"]] = relationship(
        "SqlRegisteredModelTag", back_populates="registered_model", cascade="all"
    )
    """
    Registered model Tags reference to SqlRegisteredModelTag.
    """

    model_versions: Mapped[List["SqlModelVersion"]] = relationship(
        "SqlModelVersion", back_populates="registered_model", cascade="all"
    )
    """
    ModelVersions reference to SqlRegisteredModel
    """

    __table_args__ = (PrimaryKeyConstraint("name", name="model_pk"),)

    def __repr__(self):
        return (
            f"<SqlRegisteredModel ({self.name}, {self.creation_time}, {self.last_updated_time},"
            f" {self.description})>"
        )

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.RegisteredModel`.
        """
        return RegisteredModel(
            name=self.name,
            creation_time=self.creation_time,
            last_updated_time=self.last_updated_time,
            description=self.description,
            tags=[tag.to_submarine_entity() for tag in self.tags],
        )


# +----------+-----------+
# | name     | tag       |
# +----------+-----------+
# | ResNet50 | image     |
# | ResNet50 | marketing |
# | BERT     | text      |
# +----------+-----------+


class SqlRegisteredModelTag(Base):
    __tablename__ = "registered_model_tag"

    name = Column(
        String(256), ForeignKey("registered_model.name", onupdate="cascade", ondelete="cascade")
    )
    """
    Name of registered model: Part of *Primary Key* for ``registered_model_tag`` table.
                              Refer to name of ``registered_model`` table.
    """

    tag = Column(String(256), nullable=False)
    """
    Registered model tag: `String` (limit 256 characters).
                          Part of *Primary Key* for ``registered_model_tag`` table.
    """

    # linked entities
    registered_model: SqlRegisteredModel = relationship("SqlRegisteredModel", back_populates="tags")

    __table_args__ = (PrimaryKeyConstraint("name", "tag", name="registered_model_tag_pk"),)

    def __repr__(self):
        return f"<SqlRegisteredModelTag ({self.name}, {self.tag})>"

    # entity mappers
    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.RegisteredModelTag`.
        """
        return RegisteredModelTag(self.tag)


# +----------+---------+----------------------------------+-----+
# | name     | version |                id                | ... |
# +----------+---------+----------------------------------+-----+
# | ResNet50 | 1       | 4ed6572b74a54020b0987ebf53170940 | ... |
# | ResNet50 | 2       | 1a67f138c1ff41778edf83451d5fd52f | ... |
# | BERT     | 1       | 42ae7f58ba354872a95f6872e16c3544 | ... |
# +----------+---------+----------------------------------+-----+


class SqlModelVersion(Base):
    __tablename__ = "model_version"

    name = Column(
        String(256),
        ForeignKey("registered_model.name", onupdate="cascade", ondelete="cascade"),
        nullable=False,
    )
    """
    Name of model version: Part of *Primary Key* for ``model_version`` table.
    """

    version = Column(Integer, nullable=False)
    """
    Version of registered model: Part of *Primary Key* for ``model_version`` table.
    """

    id = Column(String(64), nullable=False)
    """
    ID of the model.
    """

    user_id = Column(String(64), nullable=False)
    """
    ID to whom this model is created.
    """

    experiment_id = Column(String(64), nullable=False)
    """
    ID to which this version of model belongs to.
    """

    model_type = Column(String(64), nullable=False)
    """
    Type of model.
    """

    current_stage = Column(String(64), default=STAGE_NONE)
    """
    Current stage of this version of model: it can be `None`, `Developing`,
                                            `Production` and `Achieved`
    """

    creation_time = Column(DATETIME(fsp=3), default=datetime.now())
    """
    Creation time of this version of model: default current time in milliseconds
    """

    last_updated_time = Column(DATETIME(fsp=3), nullable=True)
    """
    Last updated time of this version of model.
    """

    dataset = Column(String(256), nullable=True, default=None)
    """
    Dataset used for this version of model.
    """

    description = Column(String(5000), nullable=True)
    """
    Description for this version of model.
    """

    tags: Mapped[List["SqlModelVersionTag"]] = relationship(
        "SqlModelVersionTag", back_populates="model_version", cascade="all"
    )
    """
    Model version tags reference to SqlModelVersionTag.
    """

    # linked entities
    registered_model: SqlRegisteredModel = relationship(
        "SqlRegisteredModel", back_populates="model_versions"
    )

    __table_args__ = (
        PrimaryKeyConstraint("name", "version", name="model_version_pk"),
        UniqueConstraint("name", "id"),
    )

    def __repr__(self):
        return (
            f"<SqlModelVersion ({self.name}, {self.version}, {self.user_id},"
            f" {self.experiment_id}, {self.current_stage}, {self.creation_time},"
            f" {self.last_updated_time}, {self.dataset}, {self.description})>"
        )

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.ModelVersion`.
        """
        return ModelVersion(
            name=self.name,
            version=self.version,
            id=self.id,
            user_id=self.user_id,
            experiment_id=self.experiment_id,
            model_type=self.model_type,
            current_stage=self.current_stage,
            creation_time=self.creation_time,
            last_updated_time=self.last_updated_time,
            dataset=self.dataset,
            description=self.description,
            tags=[tag.to_submarine_entity() for tag in self.tags],
        )


# +----------+---------+----------+
# | name     | version | tag      |
# +----------+---------+----------+
# | ResNet50 | 1       | best     |
# | ResNet50 | 1       | serving  |
# | ResNet50 | 2       | new      |
# | BERT     | 1       | testing  |
# +----------+---------+----------+


class SqlModelVersionTag(Base):
    __tablename__ = "model_version_tag"

    name = Column(String(256), nullable=False)
    """
    Name of registered model: Part of *Foreign Key* for ``model_version_tag`` table.
                              Refer to name of ``model_version`` table.
    """

    version = Column(Integer, nullable=False)
    """
    version of model: Part of *Foreign Key* for ``model_version_tag`` table.
                      Refer to version of ``model_version`` table.
    """

    tag = Column(String(256), nullable=False)
    """
    tag of model version: `String` (limit 256 characters).
                          Part of *Primary Key* for ``model_tag`` table.
    """

    # linked entities
    model_version: SqlModelVersion = relationship(
        "SqlModelVersion", foreign_keys=[name, version], back_populates="tags"
    )

    __table_args__ = (
        PrimaryKeyConstraint("name", "version", "tag", name="model_version_tag_pk"),
        ForeignKeyConstraint(
            ("name", "version"),
            ("model_version.name", "model_version.version"),
            onupdate="cascade",
            ondelete="cascade",
        ),
    )

    def __repr__(self):
        return f"<SqlModelVersionTag ({self.name}, {self.version}, {self.tag})>"

    # entity mappers
    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.ModelVersionTag`.
        """
        return ModelVersionTag(self.tag)


# +--------------------+-----------------+-----------+-------------------------+-----+
# | id                 | experiment_spec | create_by | create_time             | ... |
# +--------------------+-----------------+-----------+-------------------------+-----+
# | application_123456 | ...             | root      | 2021-08-30 10:10:10.100 | ... |
# +--------------------+-----------------+-----------+-------------------------+-----+


class SqlExperiment(Base):
    __tablename__ = "experiment"

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

    __table_args__ = (PrimaryKeyConstraint("id"),)

    def __repr__(self):
        return "<SqlMetric({}, {}, {}, {}, {}, {})>".format(
            self.id,
            self.experiment_spec,
            self.create_by,
            self.create_time,
            self.update_by,
            self.update_time,
        )

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.Experiment`.
        """
        return Experiment(
            id=self.id,
            experiment_spec=self.experiment_spec,
            create_by=self.create_by,
            create_time=self.create_time,
            update_by=self.update_by,
            update_time=self.update_time,
        )


# +--------------------+-------+-------------------+--------------+-------------------------+-----+
# | id                 | key   | value             | worker_index | timestamp               | ... |
# +--------------------+-------+-------------------+--------------+-------------------------+-----+
# | application_123456 | score | 0.666666666666667 | worker-1     | 2021-08-30 10:10:10.100 | ... |
# | application_123456 | score | 0.666666668777777 | worker-1     | 2021-08-30 10:10:11.156 | ... |
# | application_123456 | score | 0.666666670000001 | worker-1     | 2021-08-30 10:10:12.201 | ... |
# +--------------------+-------+-------------------+--------------+-------------------------+-----+


class SqlMetric(Base):
    __tablename__ = "metric"

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
    __tablename__ = "param"

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
        return f"<SqlParam({self.key}, {self.value}, {self.worker_index})>"

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.Param`.
        """
        return Param(key=self.key, value=self.value, worker_index=self.worker_index)
