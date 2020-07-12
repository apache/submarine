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

import io
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from pyarrow import fs


def open_buffered_file_reader(
        uri: str,
        buffer_size: int = io.DEFAULT_BUFFER_SIZE) -> io.BufferedReader:
    try:
        input_file = open_input_file(uri)
        return io.BufferedReader(input_file, buffer_size=buffer_size)
    except Exception as e:
        input_file.close()
        raise e


def open_buffered_stream_writer(
        uri: str,
        buffer_size: int = io.DEFAULT_BUFFER_SIZE) -> io.BufferedWriter:
    try:
        output_stream = open_output_stream(uri)
        return io.BufferedWriter(output_stream, buffer_size=buffer_size)
    except Exception as e:
        output_stream.close()
        raise e


def write_file(buffer: io.BytesIO,
               uri: str,
               buffer_size: int = io.DEFAULT_BUFFER_SIZE) -> None:
    with open_buffered_stream_writer(uri,
                                     buffer_size=buffer_size) as output_stream:
        output_stream.write(buffer.getbuffer())


def open_input_file(uri: str):
    filesystem, path = _parse_uri(uri)
    return filesystem.open_input_file(path)


def open_output_stream(uri: str):
    filesystem, path = _parse_uri(uri)
    return filesystem.open_output_stream(path)


def file_info(uri: str) -> fs.FileInfo:
    filesystem, path = _parse_uri(uri)
    info, = filesystem.get_file_info([path])
    return info


def _parse_uri(uri: str) -> Tuple[fs.FileSystem, str]:
    parsed = urlparse(uri)
    uri = uri if parsed.scheme else str(
        Path(parsed.path).expanduser().absolute())
    filesystem, path = fs.FileSystem.from_uri(uri)
    return filesystem, path
