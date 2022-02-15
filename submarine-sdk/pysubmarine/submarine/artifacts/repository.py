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
    def __init__(self):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        )
        self.bucket = "submarine"

    def _upload_file(self, local_file: str, bucket: str, key: str) -> None:
        self.client.upload_file(Filename=local_file, Bucket=bucket, Key=key)

    def list_artifact_subfolder(self, dest_path):
        response = self.client.list_objects(
            Bucket=self.bucket,
            Prefix=f"{dest_path}/",
            Delimiter="/",
        )
        return response.get("CommonPrefixes")

    def log_artifact(self, dest_path: str, local_file: str) -> None:
        dest_path = os.path.join(dest_path, os.path.basename(local_file))
        self._upload_file(
            local_file=local_file,
            bucket=self.bucket,
            key=dest_path,
        )

    def log_artifacts(self, dest_path: str, local_dir: str) -> str:
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
        return f"s3://{self.bucket}/{dest_path}"

    def delete_folder(self, dest_path) -> None:
        objects_to_delete = self.client.list_objects(Bucket=self.bucket, Prefix=dest_path)
        if objects_to_delete.get("Contents") is not None:
            delete_keys: dict = {"Objects": []}
            delete_keys["Objects"] = [
                {"Key": k} for k in [obj["Key"] for obj in objects_to_delete.get("Contents")]
            ]
            self.client.delete_objects(Bucket=self.bucket, Delete=delete_keys)
