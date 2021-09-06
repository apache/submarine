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

import mysql.connector

conn = mysql.connector.connect(user="root", password="password", host="127.0.0.1")

cursor = conn.cursor(buffered=True)

assert conn.is_connected(), "mysql is not connected"


def commit(sql):
    try:
        # Executing the SQL command
        cursor.execute(sql)
        if cursor.with_rows:
            print(cursor.fetchall())
        # Commit your changes in the database
        conn.commit()

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
        # Rolling back in case of error
        conn.rollback()


def commit_from_file(file_path):
    with open(file_path) as f:
        for result in cursor.execute(f.read(), multi=True):
            if result.with_rows:
                print(result.fetchall())


commit("CREATE DATABASE IF NOT EXISTS submarine_test;")
commit("CREATE USER IF NOT EXISTS 'submarine_test'@'%' IDENTIFIED BY 'password_test';")
commit("GRANT ALL PRIVILEGES ON *.* TO 'submarine_test'@'%';")
commit("use submarine_test;")
commit_from_file("./dev-support/database/submarine.sql")
commit("show tables;")


commit("CREATE DATABASE IF NOT EXISTS metastore_test;")
commit("CREATE USER IF NOT EXISTS 'metastore_test'@'%' IDENTIFIED BY 'password_test';")
commit("GRANT ALL PRIVILEGES ON *.* TO 'metastore_test'@'%';")
commit("use metastore_test;")
commit_from_file("./dev-support/database/metastore.sql")
commit("show tables;")


commit("CREATE DATABASE IF NOT EXISTS submarine;")
commit("CREATE USER IF NOT EXISTS 'submarine'@'%' IDENTIFIED BY 'password';")
commit("GRANT ALL PRIVILEGES ON *.* TO 'submarine'@'%';")
commit("use submarine;")
commit_from_file("./dev-support/database/submarine.sql")
commit_from_file("./dev-support/database/submarine-data.sql")
commit("show tables;")


commit("CREATE DATABASE IF NOT EXISTS metastore;")
commit("CREATE USER IF NOT EXISTS 'metastore'@'%' IDENTIFIED BY 'password';")
commit("GRANT ALL PRIVILEGES ON *.* TO 'metastore'@'%';")
commit("use metastore;")
commit_from_file("./dev-support/database/metastore.sql")
commit("show tables;")
