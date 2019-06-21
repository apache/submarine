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

files = os.listdir("./")
for name in files:
    logging.info(name)
if not os.path.isfile('./common.zip'):
    logging.error('common.zip doesn\'t exist')
    exit(-1)
if not os.path.isfile('./test20.zip'):
    logging.error('test20.zip doesn\'t exist')
    exit(-1)
if not os.path.isfile('./test2.zip/123.xml'):
    logging.error('123.xml doesn\'t exist')
    exit(-1)
