# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

"""
    Submarine API

    The Submarine REST API allows you to access Submarine resources such as,  experiments, environments and notebooks. The  API is hosted under the /v1 path on the Submarine server. For example,  to list experiments on a server hosted at http://localhost:8080, access http://localhost:8080/api/v1/experiment/  # noqa: E501

    The version of the OpenAPI document: 0.8.0-SNAPSHOT
    Contact: dev@submarine.apache.org
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from submarine.client.configuration import Configuration


class ServeSpec(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'id': 'int',
        'model_id': 'str',
        'model_name': 'str',
        'model_type': 'str',
        'model_uri': 'str',
        'model_version': 'int',
    }

    attribute_map = {
        'id': 'id',
        'model_id': 'modelId',
        'model_name': 'modelName',
        'model_type': 'modelType',
        'model_uri': 'modelURI',
        'model_version': 'modelVersion',
    }

    def __init__(
        self,
        id=None,
        model_id=None,
        model_name=None,
        model_type=None,
        model_uri=None,
        model_version=None,
        local_vars_configuration=None,
    ):  # noqa: E501
        """ServeSpec - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._id = None
        self._model_id = None
        self._model_name = None
        self._model_type = None
        self._model_uri = None
        self._model_version = None
        self.discriminator = None

        if id is not None:
            self.id = id
        if model_id is not None:
            self.model_id = model_id
        if model_name is not None:
            self.model_name = model_name
        if model_type is not None:
            self.model_type = model_type
        if model_uri is not None:
            self.model_uri = model_uri
        if model_version is not None:
            self.model_version = model_version

    @property
    def id(self):
        """Gets the id of this ServeSpec.  # noqa: E501


        :return: The id of this ServeSpec.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this ServeSpec.


        :param id: The id of this ServeSpec.  # noqa: E501
        :type: int
        """

        self._id = id

    @property
    def model_id(self):
        """Gets the model_id of this ServeSpec.  # noqa: E501


        :return: The model_id of this ServeSpec.  # noqa: E501
        :rtype: str
        """
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        """Sets the model_id of this ServeSpec.


        :param model_id: The model_id of this ServeSpec.  # noqa: E501
        :type: str
        """

        self._model_id = model_id

    @property
    def model_name(self):
        """Gets the model_name of this ServeSpec.  # noqa: E501


        :return: The model_name of this ServeSpec.  # noqa: E501
        :rtype: str
        """
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        """Sets the model_name of this ServeSpec.


        :param model_name: The model_name of this ServeSpec.  # noqa: E501
        :type: str
        """

        self._model_name = model_name

    @property
    def model_type(self):
        """Gets the model_type of this ServeSpec.  # noqa: E501


        :return: The model_type of this ServeSpec.  # noqa: E501
        :rtype: str
        """
        return self._model_type

    @model_type.setter
    def model_type(self, model_type):
        """Sets the model_type of this ServeSpec.


        :param model_type: The model_type of this ServeSpec.  # noqa: E501
        :type: str
        """

        self._model_type = model_type

    @property
    def model_uri(self):
        """Gets the model_uri of this ServeSpec.  # noqa: E501


        :return: The model_uri of this ServeSpec.  # noqa: E501
        :rtype: str
        """
        return self._model_uri

    @model_uri.setter
    def model_uri(self, model_uri):
        """Sets the model_uri of this ServeSpec.


        :param model_uri: The model_uri of this ServeSpec.  # noqa: E501
        :type: str
        """

        self._model_uri = model_uri

    @property
    def model_version(self):
        """Gets the model_version of this ServeSpec.  # noqa: E501


        :return: The model_version of this ServeSpec.  # noqa: E501
        :rtype: int
        """
        return self._model_version

    @model_version.setter
    def model_version(self, model_version):
        """Sets the model_version of this ServeSpec.


        :param model_version: The model_version of this ServeSpec.  # noqa: E501
        :type: int
        """

        self._model_version = model_version

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict()) if hasattr(item[1], "to_dict") else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, ServeSpec):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, ServeSpec):
            return True

        return self.to_dict() != other.to_dict()
