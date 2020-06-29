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


class SubmarineException(Exception):
    """
    Generic exception thrown to surface failure information about external-facing operations.
    """

    def __init__(self, message):
        """
        :param message: The message describing the error that occurred.
        """
        self.message = message
        super(SubmarineException, self).__init__(message)


class RestException(SubmarineException):
    """Exception thrown on non 200-level responses from the REST API"""

    def __init__(self, json):
        error_code = json.get('error_code')
        message = "%s: %s" % (error_code, json['message'] if 'message' in json
                              else "Response: " + str(json))
        super(RestException, self).__init__(message)
        self.json = json
