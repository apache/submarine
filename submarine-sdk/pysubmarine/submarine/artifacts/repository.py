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

import os

import boto3

from .constant import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL


class Repository:

    def __init__(self, experiment_id, config=None):
        if config is None:
            self.client = self._get_s3_client()
        else:
            self.client = boto3.client(
                "s3",
                aws_access_key_id=config.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=config.get("AWS_SECRET_ACCESS_KEY"),
                endpoint_url=config.get("S3_ENDPOINT_URL"),
            )
        self.dest_path = experiment_id

    def _get_s3_client(self):
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            endpoint_url=S3_ENDPOINT_URL,
        )

    def _upload_file(self, local_file, bucket, key):
        self.client.upload_file(Filename=local_file, Bucket=bucket, Key=key)

    def _list_artifact_subfolder(self, artifact_path):
        response = self.client.list_objects(
            Bucket="submarine",
            Prefix=os.path.join(self.dest_path, artifact_path) + "/",
            Delimiter="/")
        return response.get("CommonPrefixes")

    def log_artifact(self, local_file, artifact_path):
        bucket = "submarine"
        dest_path = self.dest_path
        dest_path = os.path.join(dest_path, artifact_path)
        dest_path = os.path.join(dest_path, os.path.basename(local_file))
        self._upload_file(
            local_file=local_file,
            bucket=bucket,
            key=dest_path,
        )

    def log_artifacts(self, local_dir, artifact_path):
        bucket = "submarine"
        dest_path = self.dest_path
        list_of_subfolder = self._list_artifact_subfolder(artifact_path)
        if list_of_subfolder is None:
            artifact_path = os.path.join(artifact_path, "1")
        else:
            artifact_path = os.path.join(artifact_path,
                                         str(len(list_of_subfolder) + 1))
        dest_path = os.path.join(dest_path, artifact_path)
        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                upload_path = os.path.join(dest_path, rel_path)
            for f in filenames:
                self._upload_file(
                    local_file=os.path.join(root, f),
                    bucket=bucket,
                    key=os.path.join(upload_path, f),
                )
