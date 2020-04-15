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

from submarine.job import SubmarineJobClient
import mock
import pytest
import json


@pytest.fixture(scope="function")
def output_json_filepath():
    dummy = {'a': 200, 'b': 2, 'c': 3}
    path = '/tmp/data.json'
    with open(path, 'w') as f:
        json.dump(dummy, f)
    return path


@mock.patch('submarine.job.submarine_job_client.http_request')
class TestSubmarineJobClient:
    def test_submit_job(self, mock_http_request, output_json_filepath):
        client = SubmarineJobClient('submarine', 8080)
        mock_http_request.return_value = {'jobId': 'job_1582524742595_0040',
                                          'name': 'submarine', 'identifier': 'test'}
        response = client.submit_job(output_json_filepath)

        with open(output_json_filepath) as json_file:
            json_body = json.load(json_file)

        mock_http_request.assert_called_with('http://submarine:8080',
                                             json_body=json_body,
                                             endpoint='/api/v1/jobs', method='POST')

        assert response['jobId'] == 'job_1582524742595_0040'
        assert response['name'] == 'submarine'
        assert response['identifier'] == 'test'

    def test_delete_job(self, mock_http_request):
        client = SubmarineJobClient('submarine', 8080)
        client.delete_job('job_1582524742595_004')
        mock_http_request.assert_called_with('http://submarine:8080',
                                             json_body=None,
                                             endpoint='/api/v1/jobs/job_1582524742595_004',
                                             method='DELETE')
