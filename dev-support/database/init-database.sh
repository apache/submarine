#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"

mysql -e "CREATE DATABASE IF NOT EXISTS submarine_test;"
mysql -e "CREATE USER IF NOT EXISTS 'submarine_test'@'%' IDENTIFIED BY 'password_test';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'submarine_test'@'%';"
mysql -e "use submarine_test; source ./submarine.sql; show tables;"

mysql -e "CREATE DATABASE IF NOT EXISTS metastore_test;"
mysql -e "CREATE USER IF NOT EXISTS 'metastore_test'@'%' IDENTIFIED BY 'password_test';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'metastore_test'@'%';"
mysql -e "use metastore_test; source ./metastore.sql; show tables;"

mysql -e "CREATE DATABASE IF NOT EXISTS submarine;"
mysql -e "CREATE USER IF NOT EXISTS 'submarine'@'%' IDENTIFIED BY 'password';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'submarine'@'%';"
mysql -e "use submarine; source ./submarine.sql; source ./submarine-data.sql; show tables;"

mysql -e "CREATE DATABASE IF NOT EXISTS metastore;"
mysql -e "CREATE USER IF NOT EXISTS 'metastore'@'%' IDENTIFIED BY 'password';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'metastore'@'%';"
mysql -e "use metastore; source ./metastore.sql; show tables;"

mysql -e "CREATE DATABASE IF NOT EXISTS mlflow;"
mysql -e "CREATE USER IF NOT EXISTS 'mlflow'@'%' IDENTIFIED BY 'password';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'mlflow'@'%';"
mysql -e "use mlflow; source ./mlflow.sql; show tables;"