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

# Install mariadb
apt-get -y install mariadb-server
service mysql start
mysql -e "CREATE DATABASE submarine_test;"
mysql -e "CREATE USER 'submarine_test'@'%' IDENTIFIED BY 'password_test';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'submarine_test'@'%';"
mysql -e "use submarine_test; source /home/yarn/database/submarine.sql; source /home/yarn/database/submarine-data.sql;"

mysql -e "CREATE DATABASE submarine;"
mysql -e "CREATE USER 'submarine'@'%' IDENTIFIED BY 'password';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'submarine'@'%';"
mysql -e "use submarine; source /home/yarn/database/submarine.sql; source /home/yarn/database/submarine-data.sql;"

mysql -e "CREATE DATABASE metastore_test;"
mysql -e "CREATE USER 'metastore_test'@'%' IDENTIFIED BY 'password_test';"
mysql -e "GRANT ALL PRIVILEGES ON * . * TO 'metastore_test'@'%';"
mysql -e "use metastore_test; source /home/yarn/database/metastore.sql;"

mysql -e "CREATE DATABASE metastore;"
mysql -e "CREATE USER 'metastore'@'%' IDENTIFIED BY 'password';"
mysql -e "GRANT ALL PRIVILEGES ON * . * TO 'metastore'@'%';"
mysql -e "use metastore; source /home/yarn/database/metastore.sql;"

mysql -e "CREATE DATABASE mlflow;"
mysql -e "CREATE USER 'mlflow'@'%' IDENTIFIED BY 'password';"
mysql -e "GRANT ALL PRIVILEGES ON * . * TO 'mlflow'@'%';"
mysql -e "use mlflow; source /home/yarn/database/mlflow.sql;"