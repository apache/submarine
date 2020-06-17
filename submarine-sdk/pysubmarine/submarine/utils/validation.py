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
"""
Utilities for validating user inputs such as metric names and parameter names.
"""
import numbers
import posixpath
import re

from submarine.exceptions import SubmarineException
from submarine.store.database.db_types import DATABASE_ENGINES

_VALID_PARAM_AND_METRIC_NAMES = re.compile(r"^[/\w.\- ]*$")

MAX_ENTITY_KEY_LENGTH = 250
MAX_PARAM_VAL_LENGTH = 250

_BAD_CHARACTERS_MESSAGE = (
    "Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),"
    " spaces ( ), and slashes (/).")

_UNSUPPORTED_DB_TYPE_MSG = "Supported database engines are {%s}" % ', '.join(
    DATABASE_ENGINES)


def bad_path_message(name):
    return (
        "Names may be treated as files in certain cases, and must not resolve to other names"
        " when treated as such. This name would resolve to '%s'"
    ) % posixpath.normpath(name)


def path_not_unique(name):
    norm = posixpath.normpath(name)
    return norm != name or norm == '.' or norm.startswith(
        '..') or norm.startswith('/')


def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise SubmarineException(
            "Invalid parameter name: '%s'. %s" %
            (name, _BAD_CHARACTERS_MESSAGE),)

    if path_not_unique(name):
        raise SubmarineException("Invalid parameter name: '%s'. %s" %
                                 (name, bad_path_message(name)))


def _validate_metric_name(name):
    """Check that `name` is a valid metric name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise SubmarineException(
            "Invalid metric name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE),)

    if path_not_unique(name):
        raise SubmarineException("Invalid metric name: '%s'. %s" %
                                 (name, bad_path_message(name)))


def _validate_length_limit(entity_name, limit, value):
    if len(value) > limit:
        raise SubmarineException(
            "%s '%s' had length %s, which exceeded length limit of %s" %
            (entity_name, value[:250], len(value), limit))


def validate_metric(key, value, timestamp, step):
    """
    Check that a param with the specified key, value, timestamp is valid and raise an exception if
    it isn't.
    """
    _validate_metric_name(key)
    if not isinstance(value, numbers.Number):
        raise SubmarineException(
            "Got invalid value %s for metric '%s' (timestamp=%s). Please specify value as a valid "
            "double (64-bit floating point)" % (value, key, timestamp),)

    if not isinstance(timestamp, numbers.Number) or timestamp < 0:
        raise SubmarineException(
            "Got invalid timestamp %s for metric '%s' (value=%s). Timestamp must be a nonnegative "
            "long (64-bit integer) " % (timestamp, key, value),)

    if not isinstance(step, numbers.Number):
        raise SubmarineException(
            "Got invalid step %s for metric '%s' (value=%s). Step must be a valid long "
            "(64-bit integer)." % (step, key, value),)


def validate_param(key, value):
    """
    Check that a param with the specified key & value is valid and raise an exception if it
    isn't.
    """
    _validate_param_name(key)
    _validate_length_limit("Param key", MAX_ENTITY_KEY_LENGTH, key)
    _validate_length_limit("Param value", MAX_PARAM_VAL_LENGTH, str(value))


def _validate_db_type_string(db_type):
    """validates db_type parsed from DB URI is supported"""
    if db_type not in DATABASE_ENGINES:
        error_msg = "Invalid database engine: '%s'. '%s'" % (
            db_type, _UNSUPPORTED_DB_TYPE_MSG)
        raise SubmarineException(error_msg)
