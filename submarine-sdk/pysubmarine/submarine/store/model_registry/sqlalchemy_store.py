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
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Union

import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import Session, sessionmaker
from sqlalchemy.orm.strategy_options import _UnboundLoad

from submarine.entities.model_registry import ModelVersion, RegisteredModel
from submarine.entities.model_registry.model_stages import (
    STAGE_DELETED_INTERNAL,
    get_canonical_stage,
)
from submarine.exceptions import SubmarineException
from submarine.store.database.models import (
    Base,
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelTag,
)
from submarine.store.model_registry.abstract_store import AbstractStore
from submarine.utils import extract_db_type_from_uri
from submarine.utils.validation import (
    validate_description,
    validate_model_name,
    validate_model_version,
    validate_tag,
    validate_tags,
)

_logger = logging.getLogger(__name__)


class SqlAlchemyStore(AbstractStore):
    def __init__(self, db_uri: str) -> None:
        """
        Create a database backed store.
        :param db_uri: The SQLAlchemy database URI string to connect to the database. See
                       the `SQLAlchemy docs
                       <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
                       for format specifications. Submarine supports the dialects ``mysql``.
        """
        super().__init__()

        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = sqlalchemy.create_engine(db_uri, pool_pre_ping=True)
        insp = sqlalchemy.inspect(self.engine)

        # Verify that all model registry tables exist.
        expected_tables = {
            SqlRegisteredModel.__tablename__,
            SqlRegisteredModelTag.__tablename__,
            SqlModelVersion.__tablename__,
            SqlModelVersionTag.__tablename__,
        }
        if len(expected_tables & set(insp.get_table_names())) == 0:
            SqlAlchemyStore._initialize_tables(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = self._get_managed_session_maker(SessionMaker)

    @staticmethod
    def _initialize_tables(engine: Engine):
        _logger.info("Creating initial Submarine database tables...")
        Base.metadata.create_all(engine)

    @staticmethod
    def _get_managed_session_maker(SessionMaker: sessionmaker):
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
            session: Session = SessionMaker()
            try:
                yield session
                session.commit()
            except SubmarineException:
                session.rollback()
                raise
            except Exception as e:
                session.rollback()
                raise SubmarineException(e)
            finally:
                session.close()

        return make_managed_session

    @staticmethod
    def _get_eager_registered_model_query_options() -> List[_UnboundLoad]:
        """
        :return A list of SQLAlchemy query options that can be used to eagerly
                load the following registered model attributes
                when fetching a model: ``registered_model_tag``.
        """
        return [sqlalchemy.orm.subqueryload(SqlRegisteredModel.tags)]

    @staticmethod
    def _get_eager_model_version_query_options():
        """
        :return: A list of SQLAlchemy query options that can be used to eagerly
                 load the following model version attributes
                 when fetching a model: ``model_version_tag``.
        """
        return [sqlalchemy.orm.subqueryload(SqlModelVersion.tags)]

    def _save_to_db(self, session: Session, objs: Union[list, object]) -> None:
        """
        Store in db
        """
        if type(objs) is list:
            session.add_all(objs)
        else:
            # single object
            session.add(objs)

    def create_registered_model(
        self, name: str, description: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> RegisteredModel:
        """
        Create a new registered model in backend store.
        :param name: Name of the new registered model.
                     This is expected to be unique in the backend store.
        :param description: Description of the registered model.
        :param tags: A list of tags associated with this registered model.
        :return: A single object of :py:class:`submarine.entities.model_registry.RegisteredModel`
                 created in the backend.
        """
        validate_model_name(name)
        validate_tags(tags)
        validate_description(description)
        with self.ManagedSessionMaker() as session:
            try:
                creation_time = datetime.now()
                registered_model = SqlRegisteredModel(
                    name=name,
                    creation_time=creation_time,
                    last_updated_time=creation_time,
                    description=description,
                    tags=[SqlRegisteredModelTag(tag=tag) for tag in tags or []],
                )
                self._save_to_db(session, registered_model)
                session.flush()
                return registered_model.to_submarine_entity()
            except sqlalchemy.exc.IntegrityError as e:
                raise SubmarineException(f"Registered model (name={name}) already exists.\nError: {str(e)}")

    @classmethod
    def _get_sql_registered_model(
        cls, session: Session, name: str, eager: bool = False
    ) -> SqlRegisteredModel:
        """
        :param eager: If ``True``, eagerly loads the registered model's tags.
                      If ``False``, these attributes are not eagerly loaded and
                      will be loaded when their corresponding object properties
                      are accessed from the resulting ``SqlRegisteredModel`` object.
        """
        validate_model_name(name)
        query_options = cls._get_eager_registered_model_query_options() if eager else []
        models: List[SqlRegisteredModel] = (
            session.query(SqlRegisteredModel)
            .options(*query_options)
            .filter(SqlRegisteredModel.name == name)
            .all()
        )

        if len(models) == 0:
            raise SubmarineException(f"Registered model with name={name} not found")
        elif len(models) > 1:
            raise SubmarineException(
                f"Expected only 1 registered model with name={name}.\nFound {len(models)}"
            )
        else:
            return models[0]

    def update_registered_model_description(self, name: str, description: str) -> RegisteredModel:
        """
        Update description of the registered model.
        :param name: Registered model name.
        :param description: New description.
        :return: A single updated :py:class:`submarine.entities.model_registry.RegisteredModel`
                 object.
        """
        validate_description(description)
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_sql_registered_model(session, name)
            sql_registered_model.description = description
            sql_registered_model.last_updated_time = datetime.now()
            self._save_to_db(session, sql_registered_model)
            session.flush()
            return sql_registered_model.to_submarine_entity()

    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """
        Rename the registered model.
        :param name: Registered model name.
        :param new_name: New proposed name.
        :return: A single updated :py:class:`submarine.entities.model_registry.RegisteredModel`
                 object.
        """
        validate_model_name(new_name)
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_sql_registered_model(session, name)
            try:
                update_time = datetime.now()
                sql_registered_model.name = new_name
                sql_registered_model.last_updated_time = update_time
                for sql_model_version in sql_registered_model.model_versions:
                    sql_model_version.name = new_name
                    sql_model_version.last_updated_time = update_time
                self._save_to_db(session, [sql_registered_model] + sql_registered_model.model_versions)
                session.flush()
                return sql_registered_model.to_submarine_entity()
            except sqlalchemy.exc.IntegrityError as e:
                raise SubmarineException(f"Registered Model (name={name}) already exists. Error: {str(e)}")

    def delete_registered_model(self, name: str) -> None:
        """
        Delete the registered model.
        :param name: Registered model name.
        :return: None.
        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_sql_registered_model(session, name)
            session.delete(sql_registered_model)

    def list_registered_model(
        self, filter_str: Optional[str] = None, filter_tags: Optional[List[str]] = None
    ) -> List[RegisteredModel]:
        """
        List of all models.
        :param filter_string: Filter query string, defaults to searching all registered models.
        :param filter_tags: Filter tags, defaults not to filter any tags.
        :return: A List of :py:class:`submarine.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions.
        """
        conditions = []
        if filter_tags is not None:
            conditions += [
                SqlRegisteredModel.tags.any(SqlRegisteredModelTag.tag.contains(tag)) for tag in filter_tags
            ]
        if filter_str is not None:
            conditions.append(SqlRegisteredModel.name.startswith(filter_str))
        with self.ManagedSessionMaker() as session:
            sql_registered_models = session.query(SqlRegisteredModel).filter(*conditions).all()
            return [
                sql_registered_model.to_submarine_entity() for sql_registered_model in sql_registered_models
            ]

    def get_registered_model(self, name: str) -> RegisteredModel:
        """
        Get registered model instance by name.
        :param name: Registered model name.
        :return: A single :py:class:`submarine.entities.model_registry.RegisteredModel` object.
        """
        with self.ManagedSessionMaker() as session:
            return self._get_sql_registered_model(session, name, True).to_submarine_entity()

    @classmethod
    def _get_registered_model_tag(cls, session: Session, name: str, tag: str) -> SqlRegisteredModelTag:
        tags = (
            session.query(SqlRegisteredModelTag)
            .filter(SqlRegisteredModelTag.name == name, SqlRegisteredModelTag.tag == tag)
            .all()
        )
        if len(tags) == 0:
            raise SubmarineException(f"Registered model tag with name={name}, tag={tag} not found")
        elif len(tags) > 1:
            raise SubmarineException(
                f"Expected only 1 registered model version tag with name={name}, tag={tag}. Found"
                f" {len(tags)}."
            )
        else:
            return tags[0]

    def add_registered_model_tag(self, name: str, tag: str) -> None:
        """
        Add a tag for the registered model.
        :param name: registered model name.
        :param tag: String of tag value.
        :return: None.
        """
        validate_model_name(name)
        validate_tag(tag)
        with self.ManagedSessionMaker() as session:
            # check if registered model exists
            self._get_sql_registered_model(session, name)
            session.merge(SqlRegisteredModelTag(name=name, tag=tag))

    def delete_registered_model_tag(self, name: str, tag: str) -> None:
        """
        Delete a tag associated with the registered model.
        :param name: Model name.
        :param tag: String of tag value.
        :return: None.
        """
        validate_model_name(name)
        validate_tag(tag)
        with self.ManagedSessionMaker() as session:
            # check if registered model exists
            self._get_sql_registered_model(session, name)
            existing_tag = self._get_registered_model_tag(session, name, tag)
            session.delete(existing_tag)

    def create_model_version(
        self,
        name: str,
        id: str,
        user_id: str,
        experiment_id: str,
        model_type: str,
        dataset: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """
        Create a new version of the registered model
        :param name: Registered model name.
        :param id: Model ID generated when model is created and stored in the description.json
        :param user_id: User ID from server that created this model
        :param experiment_id: Experiment ID which this model is created.
        :param dataset: Dataset which this version of model is used.
        :param description: Description of this version.
        :param tags: A list of string associated with this version of model.
        :return: A single object of :py:class:`submarine.entities.model_registry.ModelVersion`
                 created in the backend.
        """

        def next_version(sql_registered_model: SqlRegisteredModel) -> int:
            if sql_registered_model.model_versions:
                return (
                    max(0 if m.version is None else m.version for m in sql_registered_model.model_versions)
                    + 1
                )
            else:
                return 1

        validate_model_name(name)
        validate_description(description)
        validate_tags(tags)
        with self.ManagedSessionMaker() as session:
            try:
                creation_time = datetime.now()
                sql_registered_model = self._get_sql_registered_model(session, name)
                sql_registered_model.last_updated_time = creation_time
                model_version = SqlModelVersion(
                    name=name,
                    version=next_version(sql_registered_model),
                    id=id,
                    user_id=user_id,
                    experiment_id=experiment_id,
                    model_type=model_type,
                    creation_time=creation_time,
                    last_updated_time=creation_time,
                    dataset=dataset,
                    description=description,
                    tags=[SqlModelVersionTag(tag=tag) for tag in tags or []],
                )
                self._save_to_db(session, [sql_registered_model, model_version])
                session.flush()
                return model_version.to_submarine_entity()
            except sqlalchemy.exc.IntegrityError:
                raise SubmarineException(f"Model create error (name={name}).")

    @classmethod
    def _get_sql_model_version(
        cls, session: Session, name: str, version: int, eager: bool = False
    ) -> SqlModelVersion:
        """
        :param eager: If ``True``, eagerly loads the model's tags.
                      If ``False``, these attributes are not eagerly loaded and
                      will be loaded when their corresponding object properties
                      are accessed from the resulting ``SqlModelVersion`` object.
        """
        validate_model_name(name)
        validate_model_version(version)
        query_options = cls._get_eager_model_version_query_options() if eager else []
        conditions = [
            SqlModelVersion.name == name,
            SqlModelVersion.version == version,
            SqlModelVersion.current_stage != STAGE_DELETED_INTERNAL,
        ]

        models: List[SqlModelVersion] = (
            session.query(SqlModelVersion).options(*query_options).filter(*conditions).all()
        )
        if len(models) == 0:
            raise SubmarineException(f"Model Version (name={name}, version={version}) not found.")
        elif len(models) > 1:
            raise SubmarineException(
                f"Expected only 1 model version with (name={name}, version={version}). Found {len(models)}."
            )
        else:
            return models[0]

    def update_model_version_description(self, name: str, version: int, description: str) -> ModelVersion:
        """
        Update description associated with the version of model in backend.
        :param name: Registered model name.
        :param version: Version of the registered model.
        :param description: New model description.
        :return: A single :py:class:`submarine.entities.model_registry.ModelVersion` object.
        """
        validate_description(description)
        with self.ManagedSessionMaker() as session:
            update_time = datetime.now()
            sql_model = self._get_sql_model_version(session, name, version)
            sql_model.description = description
            sql_model.last_updated_time = update_time
            self._save_to_db(session, sql_model)
            return sql_model.to_submarine_entity()

    def transition_model_version_stage(self, name: str, version: int, stage: str) -> ModelVersion:
        """
        Update this version's stage.
        :param name: Registered model name.
        :param version: Version of the registered model.
        :param stage: New desired stage for this version of registered model.
        :return: A single :py:class:`submarine.entities.model_registry.ModelVersion` object.
        """
        with self.ManagedSessionMaker() as session:
            last_updated_time = datetime.now()

            sql_model_version = self._get_sql_model_version(session, name, version)
            sql_model_version.current_stage = get_canonical_stage(stage)
            sql_model_version.last_updated_time = last_updated_time
            sql_registered_model = sql_model_version.registered_model
            sql_registered_model.last_updated_time = last_updated_time
            self._save_to_db(session, [sql_model_version, sql_registered_model])
            return sql_model_version.to_submarine_entity()

    def delete_model_version(self, name: str, version: int) -> None:
        """
        Delete model version in backend.
        :param name: Registered model name.
        :param version: Version of the registered model.
        :return: None
        """
        with self.ManagedSessionMaker() as session:
            updated_time = datetime.now()
            sql_model_version = self._get_sql_model_version(session, name, version)
            sql_registered_model = sql_model_version.registered_model
            sql_registered_model.last_updated_time = updated_time
            session.delete(sql_model_version)
            self._save_to_db(session, sql_registered_model)
            session.flush()

    def get_model_version(self, name: str, version: int) -> ModelVersion:
        """
        Get the model by name and version.
        :param name: Registered model name.
        :param version: Version of registered model.
        :return: A single :py:class:`submarine.entities.model_registry.ModelVersion` object.
        """
        with self.ManagedSessionMaker() as session:
            sql_model_version = self._get_sql_model_version(session, name, version, True)
            return sql_model_version.to_submarine_entity()

    def list_model_versions(self, name: str, filter_tags: Optional[list] = None) -> List[ModelVersion]:
        """
        List of all models that satisfy the filter criteria.
        :param name: Registered model name.
        :param filter_tags: Filter tags, defaults not to filter any tags.
        :return: A List of :py:class:`submarine.entities.model_registry.ModelVersion` objects
                that satisfy the search expressions.
        """
        conditions = [SqlModelVersion.name == name]
        if filter_tags is not None:
            conditions += [
                SqlModelVersion.tags.any(SqlModelVersionTag.tag.contains(tag)) for tag in filter_tags
            ]
        with self.ManagedSessionMaker() as session:
            sql_models = session.query(SqlModelVersion).filter(*conditions).all()
            return [sql_model.to_submarine_entity() for sql_model in sql_models]

    def get_model_version_uri(self, name: str, version: int) -> str:
        """
        Get the location in Model registry for this version.
        :param name: Registered model name.
        :param version: Version of registered model.
        :return: A single URI location.
        """
        with self.ManagedSessionMaker() as session:
            mv = self._get_sql_model_version(session, name, version)
            return f"s3://submarine/registry/{mv.id}/{mv.name}/{mv.version}"

    @classmethod
    def _get_sql_model_version_tag(
        cls, session: Session, name: str, version: int, tag: str
    ) -> SqlModelVersionTag:
        tags = (
            session.query(SqlModelVersionTag)
            .filter(
                SqlModelVersionTag.name == name,
                SqlModelVersionTag.name == name,
                SqlModelVersionTag.version == version,
                SqlModelVersionTag.tag == tag,
            )
            .all()
        )
        if len(tags) == 0:
            raise SubmarineException(
                f"Model version tag with name={name}, version={version}, tag={tag} not found"
            )
        elif len(tags) > 1:
            raise SubmarineException(
                f"Expected only 1 model version tag with name={name}, version={version}, tag={tag}."
                f" Found {len(tags)}."
            )
        else:
            return tags[0]

    def add_model_version_tag(self, name: str, version: int, tag: str) -> None:
        """
        Add a tag for this version of model.
        :param name: Registered model name.
        :param version: Version of registered model.
        :param tag: String of tag value.
        :return: None.
        """
        validate_model_name(name)
        validate_model_version(version)
        validate_tag(tag)
        with self.ManagedSessionMaker() as session:
            # check if model version exists
            self._get_sql_model_version(session, name, version)
            session.merge(SqlModelVersionTag(name=name, version=version, tag=tag))

    def delete_model_version_tag(self, name: str, version: int, tag: str) -> None:
        """
        Delete a tag associated with this version of model.
        :param name: Registered model name.
        :param version: Version of registered model.
        :param tag: String of tag value.
        :return: None.
        """
        validate_model_name(name)
        validate_model_version(version)
        validate_tag(tag)
        with self.ManagedSessionMaker() as session:
            # check if model version exists
            self._get_sql_model_version(session, name, version)
            existing_tag = self._get_sql_model_version_tag(session, name, version, tag)
            session.delete(existing_tag)
