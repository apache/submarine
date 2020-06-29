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
from sqlalchemy import (BigInteger, Boolean, Column, PrimaryKeyConstraint,
                        String)
from sqlalchemy.ext.declarative import declarative_base

from submarine.entities import Metric, Param

Base = declarative_base()

# +-------+----------+--------------+---------------+------+--------+------------------+
# | key   | value    | worker_index | timestamp     | step | is_nan | job_name         |
# +-------+----------+--------------+---------------+------+--------+------------------+
# | score | 0.666667 | worker-1     | 1569139525097 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569149139731 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569169376482 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569236290721 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569236466722 |    0 |      0 | application_1234 |
# +-------+----------+--------------+---------------+------+--------+------------------+


class SqlMetric(Base):
    __tablename__ = 'metrics'

    key = Column(String(190))
    """
    Metric key: `String` (limit 190 characters). Part of *Primary Key* for ``metrics`` table.
    """
    value = Column(sa.types.Float(precision=53), nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    worker_index = Column(String(32))
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
    job_name = Column(String(32))
    """
    JOB NAME to which this metric belongs to: Part of *Primary Key* for ``metrics`` table.
    """

    __table_args__ = (PrimaryKeyConstraint('key',
                                           'timestamp',
                                           'worker_index',
                                           'step',
                                           'job_name',
                                           'value',
                                           "is_nan",
                                           name='metric_pk'),)

    def __repr__(self):
        return '<SqlMetric({}, {}, {}, {}, {})>'.format(self.key, self.value,
                                                        self.worker_index,
                                                        self.timestamp,
                                                        self.step)

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding Submarine entity.
        :return: :py:class:`submarine.entities.Metric`.
        """
        return Metric(key=self.key,
                      value=self.value if not self.is_nan else float("nan"),
                      worker_index=self.worker_index,
                      timestamp=self.timestamp,
                      step=self.step)


# +----------+-------+--------------+-----------------------+
# | key      | value | worker_index | job_name              |
# +----------+-------+--------------+-----------------------+
# | max_iter | 100   | worker-1     | application_123651651 |
# | n_jobs   | 5     | worker-1     | application_123456898 |
# | alpha    | 20    | worker-1     | application_123456789 |
# +----------+-------+--------------+-----------------------+


class SqlParam(Base):
    __tablename__ = 'params'

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
    job_name = Column(String(32))
    """
    JOB NAME to which this parameter belongs to: Part of *Primary Key* for ``params`` table.
    """

    __table_args__ = (PrimaryKeyConstraint('key',
                                           'job_name',
                                           'worker_index',
                                           name='param_pk'),)

    def __repr__(self):
        return '<SqlParam({}, {}, {})>'.format(self.key, self.value,
                                               self.worker_index)

    def to_submarine_entity(self):
        """
        Convert DB model to corresponding submarine entity.
        :return: :py:class:`submarine.entities.Param`.
        """
        return Param(key=self.key,
                     value=self.value,
                     worker_index=self.worker_index)
