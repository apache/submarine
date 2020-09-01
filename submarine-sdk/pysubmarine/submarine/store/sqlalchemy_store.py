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

import logging
import math
from contextlib import contextmanager

import sqlalchemy

from submarine.exceptions import SubmarineException
from submarine.store.abstract_store import AbstractStore
from submarine.store.database.models import Base, SqlMetric, SqlParam
from submarine.utils import extract_db_type_from_uri

_logger = logging.getLogger(__name__)


class SqlAlchemyStore(AbstractStore):
    """
    SQLAlchemy compliant backend store for tracking meta data for Submarine entities. Submarine
    supports the database dialects ``mysql``.
    As specified in the
    `SQLAlchemy docs <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ ,
    the database URI is expected in the format
    ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``. If you do not
    specify a driver, SQLAlchemy uses a dialect's default driver.
    This store interacts with SQL store using SQLAlchemy abstractions defined for
    Submarine entities.
    :py:class:`submarine.store.database.models.SqlMetric`, and
    :py:class:`submarine.store.database.models.SqlParam`.
    """

    def __init__(self, db_uri):
        """
        Create a database backed store.
        :param db_uri: The SQLAlchemy database URI string to connect to the database. See
                       the `SQLAlchemy docs
                       <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
                       for format specifications. Submarine supports the dialects ``mysql``.
        """
        super(SqlAlchemyStore, self).__init__()
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = sqlalchemy.create_engine(db_uri, pool_pre_ping=True)
        insp = sqlalchemy.inspect(self.engine)

        expected_tables = {
            SqlMetric.__tablename__,
            SqlParam.__tablename__,
        }
        if len(expected_tables & set(insp.get_table_names())) == 0:
            SqlAlchemyStore._initialize_tables(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = self._get_managed_session_maker(SessionMaker)
        # TODO(pingsutw): Need to check database's schema is not out of date
        # SqlAlchemyStore._verify_schema(self.engine)

    @staticmethod
    def _initialize_tables(engine):
        _logger.info("Creating initial Submarine database tables...")
        Base.metadata.create_all(engine)

    @staticmethod
    def _get_managed_session_maker(SessionMaker):
        """
        Creates a factory for producing exception-safe SQLAlchemy sessions that are made available
        using a context manager. Any session produced by this factory is automatically committed
        if no exceptions are encountered within its associated context. If an exception is
        encountered, the session is rolled back. Finally, any session produced by this factory is
        automatically closed when the session's associated context is exited.
        """

        @contextmanager
        def make_managed_session():
            """Provide a transactional scope around a series of operations."""
            session = SessionMaker()
            try:
                yield session
                session.commit()
            except SubmarineException:
                session.rollback()
                raise
            except Exception as e:
                session.rollback()
                raise SubmarineException(message=e)
            finally:
                session.close()

        return make_managed_session

    @staticmethod
    def _save_to_db(session, objs):
        """
        Store in db
        """
        if type(objs) is list:
            session.add_all(objs)
        else:
            # single object
            session.add(objs)

    def _get_or_create(self, session, model, **kwargs):
        instance = session.query(model).filter_by(**kwargs).first()
        created = False

        if instance:
            return instance, created
        else:
            instance = model(**kwargs)
            self._save_to_db(objs=instance, session=session)
            created = True

        return instance, created

    def log_metric(self, job_id, metric):
        is_nan = math.isnan(metric.value)
        if is_nan:
            value = 0
        elif math.isinf(metric.value):
            #  NB: Sql can not represent Infs = > We replace +/- Inf with max/min 64b float value
            value = 1.7976931348623157e308 if metric.value > 0 else -1.7976931348623157e308
        else:
            # some driver doesn't knows float64, so we need convert it to a regular float
            value = float(metric.value)
        with self.ManagedSessionMaker() as session:
            try:
                self._get_or_create(model=SqlMetric,
                                    id=job_id,
                                    key=metric.key,
                                    value=value,
                                    worker_index=metric.worker_index,
                                    timestamp=metric.timestamp,
                                    step=metric.step,
                                    session=session,
                                    is_nan=is_nan)
            except sqlalchemy.exc.IntegrityError:
                session.rollback()

    def log_param(self, job_id, param):
        with self.ManagedSessionMaker() as session:
            try:
                self._get_or_create(model=SqlParam,
                                    id=job_id,
                                    session=session,
                                    key=param.key,
                                    value=param.value,
                                    worker_index=param.worker_index)
                session.commit()
            except sqlalchemy.exc.IntegrityError:
                session.rollback()
