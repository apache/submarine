# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import logging
import os
import sys

time.sleep(1)

# Set up logging.
log_root = logging.getLogger()
log_root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
log_root.addHandler(ch)

if os.environ.get('RANK') is None:
    logging.error('Failed to find RANK environment variable')
    exit(1)
if os.environ.get('WORLD') is None:
    logging.error('Failed to find WORLD environment variable')
    exit(1)
if os.environ.get('INIT_METHOD') is None:
    logging.error('Failed to find INIT_METHOD environment variable')
    exit(1)

exit(0)
