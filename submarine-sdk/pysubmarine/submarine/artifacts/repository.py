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


class Repository:
    def __init__(self, experiment_id: str):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        )
        self.dest_path = experiment_id
        self.bucket = "submarine"

    def _upload_file(self, local_file: str, bucket: str, key: str) -> None:
        self.client.upload_file(Filename=local_file, Bucket=bucket, Key=key)

    def _list_artifact_subfolder(self, artifact_path: str):
        response = self.client.list_objects(
            Bucket=self.bucket,
            Prefix=os.path.join(self.dest_path, artifact_path) + "/",
            Delimiter="/",
        )
        return response.get("CommonPrefixes")

    def log_artifact(self, local_file: str, artifact_path: str) -> None:
        dest_path = self.dest_path
        dest_path = os.path.join(dest_path, artifact_path)
        dest_path = os.path.join(dest_path, os.path.basename(local_file))
        self._upload_file(
            local_file=local_file,
            bucket=self.bucket,
            key=dest_path,
        )

    def log_artifacts(self, local_dir: str, artifact_path: str) -> str:
        dest_path = self.dest_path
        list_of_subfolder = self._list_artifact_subfolder(artifact_path)
        if list_of_subfolder is None:
            artifact_path = os.path.join(artifact_path, "1")
        else:
            artifact_path = os.path.join(artifact_path, str(len(list_of_subfolder) + 1))
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
                    bucket=self.bucket,
                    key=os.path.join(upload_path, f),
                )
        return f"s3://{bucket}/{dest_path}"

    def delete_folder(self) -> None:
        objects_to_delete = self.client.list_objects(Bucket=self.bucket, Prefix=self.dest_path)
        if objects_to_delete.get("Contents") is not None:
            delete_keys: dict = {"Objects": []}
            delete_keys["Objects"] = [
                {"Key": k} for k in [obj["Key"] for obj in objects_to_delete.get("Contents")]
            ]
            self.client.delete_objects(Bucket=self.bucket, Delete=delete_keys)
