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
from datetime import datetime
from typing import List, Optional

from submarine.exceptions import SubmarineException
from submarine.store.database.db_types import DATABASE_ENGINES

_VALID_PARAM_AND_METRIC_NAMES = re.compile(r"^[/\w.\- ]*$")

MAX_ENTITY_KEY_LENGTH = 250
MAX_PARAM_VAL_LENGTH = 250

_BAD_CHARACTERS_MESSAGE = (
    "Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),"
    " spaces ( ), and slashes (/)."
)

_UNSUPPORTED_DB_TYPE_MSG = "Supported database engines are {%s}" % ", ".join(DATABASE_ENGINES)


def bad_path_message(name: str):
    return (
        "Names may be treated as files in certain cases, and must not resolve to other names"
        " when treated as such. This name would resolve to '%s'"
        % posixpath.normpath(name)
    )


def path_not_unique(name: str):
    norm = posixpath.normpath(name)
    return norm != name or norm == "." or norm.startswith("..") or norm.startswith("/")


def _validate_param_name(name: str):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise SubmarineException(
            f"Invalid parameter name: '{name}'. {_BAD_CHARACTERS_MESSAGE}",
        )

    if path_not_unique(name):
        raise SubmarineException(f"Invalid parameter name: '{name}'. {bad_path_message(name)}")


def _validate_metric_name(name: str):
    """Check that `name` is a valid metric name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise SubmarineException(
            f"Invalid metric name: '{name}'. {_BAD_CHARACTERS_MESSAGE}",
        )

    if path_not_unique(name):
        raise SubmarineException(f"Invalid metric name: '{name}'. {bad_path_message(name)}")


def _validate_length_limit(entity_name: str, limit: int, value):
    if len(value) > limit:
        raise SubmarineException(
            "%s '%s' had length %s, which exceeded length limit of %s"
            % (entity_name, value[:250], len(value), limit)
        )


def validate_metric(key, value, timestamp, step) -> None:
    """
    Check that a param with the specified key, value, timestamp is valid and raise an exception if
    it isn't.
    """
    _validate_metric_name(key)
    if not isinstance(value, numbers.Number):
        raise SubmarineException(
            "Got invalid value %s for metric '%s' (timestamp=%s). Please specify value as a valid "
            "double (64-bit floating point)" % (value, key, timestamp),
        )

    if not isinstance(timestamp, datetime):
        raise SubmarineException(
            f"Got invalid timestamp {timestamp} for metric '{key}' (value={value}). Timestamp must be a"
            " datetime object."
        )

    if not isinstance(step, numbers.Number):
        raise SubmarineException(
            f"Got invalid step (step) for metric '{key}' (value={value}). Step must be a valid long (64-bit"
            " integer)."
        )


def validate_param(key, value) -> None:
    """
    Check that a param with the specified key & value is valid and raise an exception if it
    isn't.
    """
    _validate_param_name(key)
    _validate_length_limit("Param key", MAX_ENTITY_KEY_LENGTH, key)
    _validate_length_limit("Param value", MAX_PARAM_VAL_LENGTH, str(value))


def validate_tags(tags: Optional[List[str]]) -> None:
    if tags is not None and not isinstance(tags, list):
        raise SubmarineException("parameter tags must be list or None.")
    for tag in tags or []:
        validate_tag(tag)


def validate_tag(tag: str) -> None:
    """Check that `tag` is a valid tag value and raise an exception if it isn't."""
    # Reuse param & metric check.
    if tag is None or tag == "":
        raise SubmarineException("Tag cannot be empty.")
    if not _VALID_PARAM_AND_METRIC_NAMES.match(tag):
        raise SubmarineException(f"Invalid tag name: '{tag}'. {_BAD_CHARACTERS_MESSAGE}")


def validate_model_name(name: str) -> None:
    if name is None or name == "":
        raise SubmarineException("Model name cannot be empty.")


def validate_model_version(version: int) -> None:
    if not isinstance(version, int):
        raise SubmarineException(f"Model version must be an integer, got {type(version)} type.")
    elif version < 1:
        raise SubmarineException(f"Model version must bigger than 0, but got {version}")


def validate_description(description: Optional[str]) -> None:
    if not isinstance(description, str) and description is not None:
        raise SubmarineException(f"Description must be String or None, but got {type(description)}")
    if isinstance(description, str) and len(description) > 5000:
        raise SubmarineException(f"Description must less than 5000 words, but got {len(description)}")


def _validate_db_type_string(db_type):
    """validates db_type parsed from DB URI is supported"""
    if db_type not in DATABASE_ENGINES:
        raise SubmarineException(f"Invalid database engine: '{db_type}'. '{_UNSUPPORTED_DB_TYPE_MSG}'")
