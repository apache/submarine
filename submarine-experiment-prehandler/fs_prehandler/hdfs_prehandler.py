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

import logging
import os
import subprocess

from fs_prehandler import FsPreHandler
from fsspec.implementations.arrow import HadoopFileSystem


class HDFSPreHandler(FsPreHandler):
    def __init__(self):
        self.hdfs_host=os.environ['HDFS_HOST']
        self.hdfs_port=os.environ['HDFS_PORT']
        self.hdfs_source=os.environ['HDFS_SOURCE']
        self.enable_kerberos=os.environ['ENABLE_KERBEROS']
        self.hadoop_home=os.environ['HADOOP_HOME']
        self.dest_minio_host=os.environ['DEST_MINIO_HOST']
        self.dest_minio_port=os.environ['DEST_MINIO_PORT']
        self.minio_access_key=os.environ['MINIO_ACCESS_KEY']
        self.minio_secert_key=os.environ['MINIO_SECRET_KEY']
        self.experiment_id=os.environ['EXPERIMENT_ID']

        logging.info('HDFS_HOST:%s' % self.hdfs_host)
        logging.info('HDFS_PORT:%s' % self.hdfs_port)
        logging.info('HDFS_SOURCE:%s' % self.hdfs_source)
        logging.info('MINIO_DEST_HOST:%s' % self.dest_minio_host)
        logging.info('MINIO_DEST_PORT:%s' % self.dest_minio_port)
        logging.info('ENABLE_KERBEROS:%s' % self.enable_kerberos)
        logging.info('EXPERIMENT_ID:%s' % self.experiment_id)

    def process(self):
        dest_path = 'submarine/experiment/' + self.experiment_id
        p = subprocess.run([self.hadoop_home+'/bin/hadoop', 'distcp'
            , '-Dfs.s3a.endpoint=http://' + self.dest_minio_host + ':' + self.dest_minio_port + '/'
            , '-Dfs.s3a.access.key=' + self.minio_access_key
            , '-Dfs.s3a.secret.key=' + self.minio_secert_key
            , '-Dfs.s3a.path.style.access=true'
            , 'hdfs://'+self.hdfs_host + ':' + self.hdfs_port + '/' + self.hdfs_source
            , 's3a://' + dest_path])

        if p.returncode == 0:
            logging.info('fetch data from hdfs://%s:%s/%s to %s complete' % (self.hdfs_host, self.hdfs_port, self.hdfs_source, dest_path))
        else:
            raise Exception( 'error occured when fetching data from hdfs://%s:%s/%s to %s' % (self.hdfs_host, self.hdfs_port, self.hdfs_source, dest_path) )
