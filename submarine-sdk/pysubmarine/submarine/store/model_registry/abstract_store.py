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

from abc import ABCMeta, abstractmethod
from typing import List

from submarine.entities.model_registry import ModelVersion, RegisteredModel


class AbstractStore:
    """
    Abstract class for Backend model registry
    This class defines the API interface for frontends to connect with various types of backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        """
        Empty constructor for now. This is deliberately not marked as abstract, else every
        derived class would be forced to create one.
        """
        pass

    @abstractmethod
    def create_registered_model(
        self, name: str, description: str = None, tags: List[str] = None
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
        pass

    @abstractmethod
    def update_registered_model_description(self, name: str, description: str) -> RegisteredModel:
        """
        Update description of the registered model.
        :param name: Registered model name.
        :param description: New description.
        :return: A single updated :py:class:`submarine.entities.model_registry.RegisteredModel`
                 object.
        """
        pass

    @abstractmethod
    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """
        Rename the registered model.
        :param name: Registered model name.
        :param new_name: New proposed name.
        :return: A single updated :py:class:`submarine.entities.model_registry.RegisteredModel`
                 object.
        """
        pass

    @abstractmethod
    def delete_registered_model(self, name: str) -> None:
        """
        Delete the registered model.
        :param name: Registered model name.
        :return: None.
        """
        pass

    @abstractmethod
    def list_registered_model(
        self, filter_str: str = None, filter_tags: List[str] = None
    ) -> List[RegisteredModel]:
        """
        List of all models.
        :param filter_string: Filter query string, defaults to searching all registered models.
        :param filter_tags: Filter tags, defaults not to filter any tags.
        :return: A List of :py:class:`submarine.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions.
        """
        pass

    @abstractmethod
    def get_registered_model(self, name: str) -> RegisteredModel:
        """
        Get registered model instance by name.
        :param name: Registered model name.
        :return: A single :py:class:`submarine.entities.model_registry.RegisteredModel` object.
        """
        pass

    @abstractmethod
    def add_registered_model_tag(self, name: str, tag: str) -> None:
        """
        Add a tag for the registered model.
        :param name: registered model name.
        :param tag: String of tag value.
        :return: None.
        """
        pass

    @abstractmethod
    def delete_registered_model_tag(self, name: str, tag: str) -> None:
        """
        Delete a tag associated with the registered model.
        :param name: Model name.
        :param tag: String of tag value.
        :return: None.
        """
        pass

    @abstractmethod
    def create_model_version(
        self,
        name: str,
        id: str,
        user_id: str,
        experiment_id: str,
        model_type: str,
        dataset: str = None,
        description: str = None,
        tags: List[str] = None,
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
        pass

    @abstractmethod
    def update_model_version_description(self, name: str, version: int, description: str) -> ModelVersion:
        """
        Update description associated with the version of model in backend.
        :param name: Registered model name.
        :param version: Version of the registered model.
        :param description: New model description.
        :return: A single :py:class:`submarine.entities.model_registry.ModelVersion` object.
        """
        pass

    @abstractmethod
    def transition_model_version_stage(self, name: str, version: int, stage: str) -> ModelVersion:
        """
        Update this version's stage.
        :param name: Registered model name.
        :param version: Version of the registered model.
        :param stage: New desired stage for this version of registered model.
        :return: A single :py:class:`submarine.entities.model_registry.ModelVersion` object.
        """

    @abstractmethod
    def delete_model_version(self, name: str, version: int) -> None:
        """
        Delete model version in backend.
        :param name: Registered model name.
        :param version: Version of the registered model.
        :return: None
        """
        pass

    @abstractmethod
    def get_model_version(self, name: str, version: int) -> ModelVersion:
        """
        Get the model by name and version.
        :param name: Registered model name.
        :param version: Version of registered model.
        :return: A single :py:class:`submarine.entities.model_registry.ModelVersion` object.
        """
        pass

    @abstractmethod
    def list_model_versions(self, name: str, filter_tags: list = None) -> List[ModelVersion]:
        """
        List of all models that satisfy the filter criteria.
        :param name: Registered model name.
        :param filter_tags: Filter tags, defaults not to filter any tags.
        :return: A List of :py:class:`submarine.entities.model_registry.ModelVersion` objects
                that satisfy the search expressions.
        """
        pass

    @abstractmethod
    def get_model_version_uri(self, name: str, version: int) -> str:
        """
        Get the location in Model registry for this version.
        :param name: Registered model name.
        :param version: Version of registered model.
        :return: A single URI location.
        """
        pass

    @abstractmethod
    def add_model_version_tag(self, name: str, version: int, tag: str) -> None:
        """
        Add a tag for this version of model.
        :param name: Registered model name.
        :param version: Version of registered model.
        :param tag: String of tag value.
        :return: None.
        """
        pass

    @abstractmethod
    def delete_model_version_tag(self, name: str, version: int, tag: str) -> None:
        """
        Delete a tag associated with this version of model.
        :param name: Registered model name.
        :param version: Version of registered model.
        :param tag: String of tag value.
        :return: None.
        """
        pass
