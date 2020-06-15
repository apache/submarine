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

from six.moves import urllib

from submarine.exceptions import SubmarineException


def extract_db_type_from_uri(db_uri):
    """
    Parse the specified DB URI to extract the database type. Confirm the database type is
    supported. If a driver is specified, confirm it passes a plausible regex.
    """
    scheme = urllib.parse.urlparse(db_uri).scheme
    scheme_plus_count = scheme.count('+')

    if scheme_plus_count == 0:
        db_type = scheme
    elif scheme_plus_count == 1:
        db_type, _ = scheme.split('+')
    else:
        error_msg = "Invalid database URI: '%s'. %s" % (db_uri,
                                                        'INVALID_DB_URI_MSG')
        raise SubmarineException(error_msg)

    return db_type
