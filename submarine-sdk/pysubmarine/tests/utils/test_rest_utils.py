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

from mock import patch, Mock
import pytest
import json
from submarine.utils.rest_utils import http_request, verify_rest_response
from submarine.exceptions import RestException, SubmarineException


def test_http_request():
    dummy_json = json.dumps({'result': {'jobId': 'job_1234567', 'name': 'submarine',
                                        'identifier': 'test'}})

    with patch('requests.request') as mock_requests:
        mock_requests.return_value.text = dummy_json
        mock_requests.return_value.status_code = 200

        result = http_request('http://submarine:8080', json_body='dummy',
                              endpoint='/api/v1/jobs', method='POST')

    assert result['jobId'] == 'job_1234567'
    assert result['name'] == 'submarine'
    assert result['identifier'] == 'test'


def test_verify_rest_response():
    # Test correct response
    mock_response = Mock()
    mock_response.status_code = 200
    verify_rest_response(mock_response, '/api/v1/jobs')

    # Test response status code not equal 200(OK) and response can parse as JSON
    mock_response.status_code = 400
    mock_json_body = {'a': 200, 'b': 2, 'c': 3}
    dummy_json = json.dumps(mock_json_body)
    mock_response.text = dummy_json

    with pytest.raises(RestException, match=str(json.loads(dummy_json))):
        verify_rest_response(mock_response, '/api/v1/jobs')

    # Test response status code not equal 200(OK) and response can not parse as JSON
    mock_json_body = 'test, 123'
    mock_response.text = mock_json_body
    with pytest.raises(SubmarineException, match='API request to endpoint /api/v1/jobs failed '
                                                 'with error code 400 != 200'):
        verify_rest_response(mock_response, '/api/v1/jobs')
