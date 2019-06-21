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
import os


tb_port = None
if 'TB_PORT' in os.environ:
    tb_port = os.environ['TB_PORT']

job_name = os.environ['JOB_NAME']

print('TB_PORT is ' + str(tb_port))
print('JOB_NAME is ' + job_name)

if tb_port and job_name != 'chief':
    raise ValueError