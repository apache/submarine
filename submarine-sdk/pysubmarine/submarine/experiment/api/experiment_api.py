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
    Submarine Experiment API

    The Submarine REST API allows you to create, list, and get experiments. The API is hosted under the /v1/experiment route on the Submarine server. For example, to list experiments on a server hosted at http://localhost:8080, access http://localhost:8080/api/v1/experiment/  # noqa: E501

    The version of the OpenAPI document: 0.6.0-SNAPSHOT
    Contact: dev@submarine.apache.org
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six

from submarine.experiment.api_client import ApiClient
from submarine.experiment.exceptions import ApiTypeError, ApiValueError  # noqa: F401


class ExperimentApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def create_experiment(self, **kwargs):  # noqa: E501
        """Create an experiment  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_experiment(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param ExperimentSpec experiment_spec:
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.create_experiment_with_http_info(**kwargs)  # noqa: E501

    def create_experiment_with_http_info(self, **kwargs):  # noqa: E501
        """Create an experiment  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_experiment_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param ExperimentSpec experiment_spec:
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["experiment_spec"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method create_experiment" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if "experiment_spec" in local_var_params:
            body_params = local_var_params["experiment_spec"]
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # HTTP header `Content-Type`
        header_params["Content-Type"] = self.api_client.select_header_content_type(  # noqa: E501
            ["application/yaml", "application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment",
            "POST",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def delete_experiment(self, id, **kwargs):  # noqa: E501
        """Delete the experiment  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_experiment(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.delete_experiment_with_http_info(id, **kwargs)  # noqa: E501

    def delete_experiment_with_http_info(self, id, **kwargs):  # noqa: E501
        """Delete the experiment  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_experiment_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["id"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method delete_experiment" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]
        # verify the required parameter 'id' is set
        if self.api_client.client_side_validation and (
            "id" not in local_var_params or local_var_params["id"] is None  # noqa: E501
        ):  # noqa: E501
            raise ApiValueError(
                "Missing the required parameter `id` when calling `delete_experiment`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in local_var_params:
            path_params["id"] = local_var_params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment/{id}",
            "DELETE",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def get_experiment(self, id, **kwargs):  # noqa: E501
        """Get the experiment's detailed info by id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_experiment(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.get_experiment_with_http_info(id, **kwargs)  # noqa: E501

    def get_experiment_with_http_info(self, id, **kwargs):  # noqa: E501
        """Get the experiment's detailed info by id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_experiment_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["id"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method get_experiment" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]
        # verify the required parameter 'id' is set
        if self.api_client.client_side_validation and (
            "id" not in local_var_params or local_var_params["id"] is None  # noqa: E501
        ):  # noqa: E501
            raise ApiValueError(
                "Missing the required parameter `id` when calling `get_experiment`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in local_var_params:
            path_params["id"] = local_var_params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment/{id}",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def get_log(self, id, **kwargs):  # noqa: E501
        """Log experiment by id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_log(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.get_log_with_http_info(id, **kwargs)  # noqa: E501

    def get_log_with_http_info(self, id, **kwargs):  # noqa: E501
        """Log experiment by id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_log_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["id"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method get_log" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]
        # verify the required parameter 'id' is set
        if self.api_client.client_side_validation and (
            "id" not in local_var_params or local_var_params["id"] is None  # noqa: E501
        ):  # noqa: E501
            raise ApiValueError(
                "Missing the required parameter `id` when calling `get_log`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in local_var_params:
            path_params["id"] = local_var_params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment/logs/{id}",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def list_experiments(self, **kwargs):  # noqa: E501
        """List experiments  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_experiments(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str status:
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.list_experiments_with_http_info(**kwargs)  # noqa: E501

    def list_experiments_with_http_info(self, **kwargs):  # noqa: E501
        """List experiments  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_experiments_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str status:
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["status"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method list_experiments" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]

        collection_formats = {}

        path_params = {}

        query_params = []
        if "status" in local_var_params and local_var_params["status"] is not None:  # noqa: E501
            query_params.append(("status", local_var_params["status"]))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def list_log(self, **kwargs):  # noqa: E501
        """List experiment's log  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_log(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str status:
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.list_log_with_http_info(**kwargs)  # noqa: E501

    def list_log_with_http_info(self, **kwargs):  # noqa: E501
        """List experiment's log  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_log_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str status:
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["status"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method list_log" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]

        collection_formats = {}

        path_params = {}

        query_params = []
        if "status" in local_var_params and local_var_params["status"] is not None:  # noqa: E501
            query_params.append(("status", local_var_params["status"]))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment/logs",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def patch_experiment(self, id, **kwargs):  # noqa: E501
        """Update the experiment in the submarine server with spec  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_experiment(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param ExperimentSpec experiment_spec:
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: JsonResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.patch_experiment_with_http_info(id, **kwargs)  # noqa: E501

    def patch_experiment_with_http_info(self, id, **kwargs):  # noqa: E501
        """Update the experiment in the submarine server with spec  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_experiment_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param str id: (required)
        :param ExperimentSpec experiment_spec:
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(JsonResponse, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ["id", "experiment_spec"]
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s' to method patch_experiment" % key
                )
            local_var_params[key] = val
        del local_var_params["kwargs"]
        # verify the required parameter 'id' is set
        if self.api_client.client_side_validation and (
            "id" not in local_var_params or local_var_params["id"] is None  # noqa: E501
        ):  # noqa: E501
            raise ApiValueError(
                "Missing the required parameter `id` when calling `patch_experiment`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in local_var_params:
            path_params["id"] = local_var_params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if "experiment_spec" in local_var_params:
            body_params = local_var_params["experiment_spec"]
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # HTTP header `Content-Type`
        header_params["Content-Type"] = self.api_client.select_header_content_type(  # noqa: E501
            ["application/yaml", "application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment/{id}",
            "PATCH",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="JsonResponse",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def ping(self, **kwargs):  # noqa: E501
        """Ping submarine server  # noqa: E501

        Return the Pong message for test the connectivity  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.ping(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        return self.ping_with_http_info(**kwargs)  # noqa: E501

    def ping_with_http_info(self, **kwargs):  # noqa: E501
        """Ping submarine server  # noqa: E501

        Return the Pong message for test the connectivity  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.ping_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool: execute request asynchronously
        :param _return_http_data_only: response data without head status code
                                       and headers
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: tuple(str, status_code(int), headers(HTTPHeaderDict))
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = []
        all_params.extend(
            ["async_req", "_return_http_data_only", "_preload_content", "_request_timeout"]
        )

        for key, val in six.iteritems(local_var_params["kwargs"]):
            if key not in all_params:
                raise ApiTypeError("Got an unexpected keyword argument '%s' to method ping" % key)
            local_var_params[key] = val
        del local_var_params["kwargs"]

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json; charset=utf-8"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            "/v1/experiment/ping",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="str",  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get("async_req"),
            _return_http_data_only=local_var_params.get("_return_http_data_only"),  # noqa: E501
            _preload_content=local_var_params.get("_preload_content", True),
            _request_timeout=local_var_params.get("_request_timeout"),
            collection_formats=collection_formats,
        )
