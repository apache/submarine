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

from submarine.store import DEFAULT_SUBMARINE_JDBC_URL
from submarine.utils import env

_DB_URI_ENV_VAR = "SUBMARINE_DB_URI"


_db_uri = None


def is_db_uri_set():
    """Returns True if the DB URI has been set, False otherwise."""
    if _db_uri or env.get_env(_DB_URI_ENV_VAR):
        return True
    return False


def set_db_uri(uri):
    """
    Set the DB URI. This does not affect the currently active run (if one exists),
    but takes effect for successive runs.
    """
    global _db_uri
    _db_uri = uri


def get_db_uri():
    """
    Get the current DB URI.
    :return: The DB URI.
    """
    global _db_uri
    if _db_uri is not None:
        return _db_uri
    elif env.get_env(_DB_URI_ENV_VAR) is not None:
        return env.get_env(_DB_URI_ENV_VAR)
    else:
        return DEFAULT_SUBMARINE_JDBC_URL
