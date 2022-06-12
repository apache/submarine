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
from fs_prehandler import FsPreHandler
from fsspec.implementations.arrow import HadoopFileSystem

class HDFSPreHandler(FsPreHandler):
    def __init__(self):
        self.hdfs_host=os.environ["HDFS_HOST"]
        self.hdfs_port=int(os.environ["HDFS_PORT"])
        self.hdfs_source=os.environ["HDFS_SOURCE"]
        self.dest_path=os.environ["DEST_PATH"]
        self.enable_kerberos=os.environ["ENABLE_KERBEROS"]

        print('HDFS_HOST:%s' % self.hdfs_host)
        print('HDFS_PORT:%d' % self.hdfs_port)
        print('HDFS_SOURCE:%s' % self.hdfs_source)
        print('DEST_PATH:%s' % self.dest_path)
        print('ENABLE_KERBEROS:%s' % self.enable_kerberos)

        self.fs = HadoopFileSystem(host=self.hdfs_host, port=self.hdfs_port)

    def process(self):
        self.fs.get(self.hdfs_source, self.dest_path, recursive=True)
        print('fetch data from hdfs://%s:%d/%s to %s complete' % (self.hdfs_host, self.hdfs_port, self.hdfs_source, self.dest_path))
