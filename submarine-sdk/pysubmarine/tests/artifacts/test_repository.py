# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pathlib
import shutil

import boto3
from moto import mock_s3

from submarine.artifacts import Repository


@mock_s3
def test_log_artifact():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="submarine")

    local_file = pathlib.Path(__file__).parent / "text.txt"
    with local_file.open("w", encoding="utf-8") as file:
        file.write("test")

    repo = Repository()
    dest_path = "folder01/subfolder01"
    repo.log_artifact(dest_path=dest_path, local_file=str(local_file))
    local_file.unlink()

    common_prefixes = repo.list_artifact_subfolder("folder01")
    assert common_prefixes == [{"Prefix": "folder01/subfolder01/"}]


@mock_s3
def test_log_artifacts():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="submarine")

    # create the following directory tree:
    # data/
    # ├── subdir-00
    # │   └── subdir-10
    # │       └── text1.txt
    # └── subdir-01
    #     └── subdir-10
    #         └── text2.txt
    local_dir = pathlib.Path(__file__).parent / "data"
    (local_dir / "subdir-00" / "subdir-10").mkdir(parents=True, exist_ok=True)
    (local_dir / "subdir-01" / "subdir-10").mkdir(parents=True, exist_ok=True)
    local_file1 = local_dir / "subdir-00" / "subdir-10" / "text1.txt"
    with local_file1.open("w", encoding="utf-8") as file:
        file.write("test")
    local_file2 = local_dir / "subdir-01" / "subdir-10" / "text2.txt"
    with local_file2.open("w", encoding="utf-8") as file:
        file.write("test")

    repo = Repository()
    s3_folder_name = repo.log_artifacts(dest_path="data", local_dir=str(local_dir))

    shutil.rmtree(local_dir)

    assert s3_folder_name == "s3://submarine/data"

    common_prefixes = repo.list_artifact_subfolder("data")
    assert common_prefixes == [{'Prefix': 'data/subdir-00/'}, {'Prefix': 'data/subdir-01/'}]


@mock_s3
def test_delete_folder():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="submarine")

    local_file = pathlib.Path(__file__).parent / "text.txt"
    with local_file.open("w", encoding="utf-8") as file:
        file.write("test")

    s3.meta.client.upload_file(str(local_file), "submarine", "folder01/subfolder01/text.txt")
    s3.meta.client.upload_file(str(local_file), "submarine", "folder01/subfolder02/text.txt")
    local_file.unlink()

    repo = Repository()
    repo.delete_folder("folder01/subfolder01")

    common_prefixes = repo.list_artifact_subfolder("folder01")
    assert common_prefixes == [{"Prefix": "folder01/subfolder02/"}]
