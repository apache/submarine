#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mysql -uroot -p$MYSQL_ROOT_PASSWORD <<EOF
CREATE DATABASE submarine_test;
CREATE USER 'submarine_test'@'%' IDENTIFIED BY 'password_test';
GRANT ALL PRIVILEGES ON *.* TO 'submarine_test'@'%';
use submarine_test; source /tmp/database/submarine.sql; source /tmp/database/submarine-model.sql;
CREATE DATABASE submarine;
CREATE USER 'submarine'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'submarine'@'%';
use submarine; source /tmp/database/submarine.sql; source /tmp/database/submarine-model.sql;
source /tmp/database/submarine-data.sql;
CREATE DATABASE metastore_test;
CREATE USER 'metastore_test'@'%' IDENTIFIED BY 'password_test';
GRANT ALL PRIVILEGES ON * . * TO 'metastore_test'@'%';
use metastore_test; source /tmp/database/metastore.sql;
CREATE DATABASE metastore;
CREATE USER 'metastore'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON * . * TO 'metastore'@'%';
use metastore; source /tmp/database/metastore.sql;
CREATE DATABASE mlflowdb;
CREATE USER 'mlflow'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON * . * TO 'mlflow'@'%';
CREATE DATABASE grafana;
CREATE USER 'grafana'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON * . * TO 'grafana'@'%';
EOF
