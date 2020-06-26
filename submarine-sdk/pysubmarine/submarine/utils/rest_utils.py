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

import json
import logging

import requests

from submarine.exceptions import RestException, SubmarineException

_logger = logging.getLogger(__name__)


def http_request(base_url,
                 endpoint,
                 method,
                 json_body,
                 timeout=60,
                 headers=None,
                 **kwargs):
    """
    Perform requests.
    :param base_url: http request base url containing hostname and port. e.g. https://submarine:8088
    :param endpoint: specified as a relative or absolute url
    :param method: http request method
    :param json_body: request json body, for `application/json`
    :param timeout: How many seconds to wait for the server to send data
    :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
    :return:
    """
    method = method.upper()
    assert method in [
        'GET', 'HEAD', 'DELETE', 'POST', 'PUT', 'PATCH', 'OPTIONS'
    ]
    headers = headers or {}
    if 'Content-Type' not in headers:
        headers['Content-Type'] = 'application/json'

    url = base_url + endpoint
    response = requests.request(url=url,
                                method=method,
                                json=json_body,
                                headers=headers,
                                timeout=timeout,
                                **kwargs)
    verify_rest_response(response, endpoint)

    response = json.loads(response.text)
    result = response['result']
    return result


def _can_parse_as_json(string):
    try:
        json.loads(string)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def verify_rest_response(response, endpoint):
    """Verify the return code and raise exception if the request was not successful."""
    if response.status_code != 200:
        if _can_parse_as_json(response.text):
            raise RestException(json.loads(response.text))
        else:
            base_msg = "API request to endpoint %s failed with error code " \
                       "%s != 200" % (endpoint, response.status_code)
            raise SubmarineException("%s. Response body: '%s'" %
                                     (base_msg, response.text))
    return response
