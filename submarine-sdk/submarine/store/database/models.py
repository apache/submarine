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
import sqlalchemy as sa
from sqlalchemy import (
    Column, String, BigInteger,
    PrimaryKeyConstraint, Boolean)
from sqlalchemy.ext.declarative import declarative_base
from submarine.entities import (Metric, Param)


Base = declarative_base()


class SqlMetric(Base):
    __tablename__ = 'metrics'

    key = Column(String(250))
    """
    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``metrics`` table.
    """
    value = Column(sa.types.Float(precision=53), nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    worker_index = Column(String(250))
    """
    Metric worker_index: `String` (limit 250 characters). Part of *Primary Key* for
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
    job_name = Column(String(32))
    """
    JOB NAME to which this metric belongs to: Part of *Primary Key* for ``metrics`` table.
    """

    __table_args__ = (
        PrimaryKeyConstraint('key', 'timestamp', 'step', 'job_name', 'value', "is_nan",
                             name='metric_pk'),
    )

    def __repr__(self):
        return '<SqlMetric({}, {}, {}, {})>'.format(self.key, self.value, self.timestamp, self.step)

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
            step=self.step)


class SqlParam(Base):
    __tablename__ = 'params'

    key = Column(String(250))
    """
    Param key: `String` (limit 250 characters). Part of *Primary Key* for ``params`` table.
    """
    value = Column(String(250), nullable=False)
    """
    Param value: `String` (limit 250 characters). Defined as *Non-null* in schema.
    """
    worker_index = Column(String(250), nullable=False)
    """
    Param worker_index: `String` (limit 250 characters). Part of *Primary Key* for
    ``metrics`` table.
    """
    job_name = Column(String(32))
    """
    JOB NAME to which this parameter belongs to: Part of *Primary Key* for ``params`` table.
    """

    __table_args__ = (
        PrimaryKeyConstraint('key', 'job_name', name='param_pk'),
    )

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.Param`.
        """
        return Param(
            key=self.key,
            value=self.value,
            worker_index=self.worker_index)
