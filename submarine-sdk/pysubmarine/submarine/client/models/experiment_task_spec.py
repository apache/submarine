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


class ExperimentTaskSpec(object):
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
        'cmd': 'str',
        'cpu': 'str',
        'env_vars': 'dict(str, str)',
        'gpu': 'str',
        'image': 'str',
        'memory': 'str',
        'name': 'str',
        'replicas': 'int',
        'resources': 'str',
    }

    attribute_map = {
        'cmd': 'cmd',
        'cpu': 'cpu',
        'env_vars': 'envVars',
        'gpu': 'gpu',
        'image': 'image',
        'memory': 'memory',
        'name': 'name',
        'replicas': 'replicas',
        'resources': 'resources',
    }

    def __init__(
        self,
        cmd=None,
        cpu=None,
        env_vars=None,
        gpu=None,
        image=None,
        memory=None,
        name=None,
        replicas=None,
        resources=None,
        local_vars_configuration=None,
    ):  # noqa: E501
        """ExperimentTaskSpec - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._cmd = None
        self._cpu = None
        self._env_vars = None
        self._gpu = None
        self._image = None
        self._memory = None
        self._name = None
        self._replicas = None
        self._resources = None
        self.discriminator = None

        if cmd is not None:
            self.cmd = cmd
        if cpu is not None:
            self.cpu = cpu
        if env_vars is not None:
            self.env_vars = env_vars
        if gpu is not None:
            self.gpu = gpu
        if image is not None:
            self.image = image
        if memory is not None:
            self.memory = memory
        if name is not None:
            self.name = name
        if replicas is not None:
            self.replicas = replicas
        if resources is not None:
            self.resources = resources

    @property
    def cmd(self):
        """Gets the cmd of this ExperimentTaskSpec.  # noqa: E501


        :return: The cmd of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._cmd

    @cmd.setter
    def cmd(self, cmd):
        """Sets the cmd of this ExperimentTaskSpec.


        :param cmd: The cmd of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._cmd = cmd

    @property
    def cpu(self):
        """Gets the cpu of this ExperimentTaskSpec.  # noqa: E501


        :return: The cpu of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._cpu

    @cpu.setter
    def cpu(self, cpu):
        """Sets the cpu of this ExperimentTaskSpec.


        :param cpu: The cpu of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._cpu = cpu

    @property
    def env_vars(self):
        """Gets the env_vars of this ExperimentTaskSpec.  # noqa: E501


        :return: The env_vars of this ExperimentTaskSpec.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._env_vars

    @env_vars.setter
    def env_vars(self, env_vars):
        """Sets the env_vars of this ExperimentTaskSpec.


        :param env_vars: The env_vars of this ExperimentTaskSpec.  # noqa: E501
        :type: dict(str, str)
        """

        self._env_vars = env_vars

    @property
    def gpu(self):
        """Gets the gpu of this ExperimentTaskSpec.  # noqa: E501


        :return: The gpu of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._gpu

    @gpu.setter
    def gpu(self, gpu):
        """Sets the gpu of this ExperimentTaskSpec.


        :param gpu: The gpu of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._gpu = gpu

    @property
    def image(self):
        """Gets the image of this ExperimentTaskSpec.  # noqa: E501


        :return: The image of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._image

    @image.setter
    def image(self, image):
        """Sets the image of this ExperimentTaskSpec.


        :param image: The image of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._image = image

    @property
    def memory(self):
        """Gets the memory of this ExperimentTaskSpec.  # noqa: E501


        :return: The memory of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._memory

    @memory.setter
    def memory(self, memory):
        """Sets the memory of this ExperimentTaskSpec.


        :param memory: The memory of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._memory = memory

    @property
    def name(self):
        """Gets the name of this ExperimentTaskSpec.  # noqa: E501


        :return: The name of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this ExperimentTaskSpec.


        :param name: The name of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def replicas(self):
        """Gets the replicas of this ExperimentTaskSpec.  # noqa: E501


        :return: The replicas of this ExperimentTaskSpec.  # noqa: E501
        :rtype: int
        """
        return self._replicas

    @replicas.setter
    def replicas(self, replicas):
        """Sets the replicas of this ExperimentTaskSpec.


        :param replicas: The replicas of this ExperimentTaskSpec.  # noqa: E501
        :type: int
        """

        self._replicas = replicas

    @property
    def resources(self):
        """Gets the resources of this ExperimentTaskSpec.  # noqa: E501


        :return: The resources of this ExperimentTaskSpec.  # noqa: E501
        :rtype: str
        """
        return self._resources

    @resources.setter
    def resources(self, resources):
        """Sets the resources of this ExperimentTaskSpec.


        :param resources: The resources of this ExperimentTaskSpec.  # noqa: E501
        :type: str
        """

        self._resources = resources

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
        if not isinstance(other, ExperimentTaskSpec):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, ExperimentTaskSpec):
            return True

        return self.to_dict() != other.to_dict()
