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
from sqlalchemy import (
    Column, String, Float,
    BigInteger, PrimaryKeyConstraint, Boolean)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

RunStatusTypes = [
    "SCHEDULED",
    "FAILED",
    "FINISHED",
    "RUNNING",
]


class SqlMetric(Base):
    __tablename__ = 'metrics'

    key = Column(String(250))
    """
    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``metrics`` table.
    """
    value = Column(Float, nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    worker_index = Column(String(250))
    """
    Param worker_index: `String` (limit 250 characters). Defined as *Non-null* in schema.
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
        return '<SqlMetric({}, {}, {})>'.format(self.key, self.value, self.timestamp)


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
    Param worker_index: `String` (limit 250 characters). Defined as *Non-null* in schema.
    """
    job_name = Column(String(32))
    """
    JOB NAME to which this metric belongs to: Part of *Primary Key* for ``params`` table.
                                              *Foreign Key* into ``runs`` table.
    """

    __table_args__ = (
        PrimaryKeyConstraint('key', 'job_name', name='param_pk'),
    )

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)
