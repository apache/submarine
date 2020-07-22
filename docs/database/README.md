<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->

# Submarine Database

Submarine needs to use the database to store information about the `organization`, `user`, `projects`, `tasks`, `metastore` and `configuration` of the system information, So consider using mysql to store this data.

+ MySQL will be included in the `mini-submarine` docker image to allow users to quickly experience the `submarine workbench`.
+ In a production environment, the `submarine workbench` can be connected to the official mysql database.

## Prerequisite

Must:

- MySQL
- MyBatis

## Run mysql on docker

By using the official docker image of submarine database, only one docker command is required to run submarine database

```bash
docker run -it -p 3306:3306 -d --name submarine-database -e MYSQL_ROOT_PASSWORD=password apache/submarine:database-0.4.0
```
## Initialize submarine database
It will create users and tables that submarine requires
```shell script
sudo ./init-database
```
## Manual operation of the submarine database

### Modify character set (Optional)

If you need to store Chinese character data in mysql, you need to execute the following command to modify the mysql character set.

+ Set database character set

  ```
  bash > mysql -uroot -ppassword

  mysql>SHOW VARIABLES LIKE 'character_set_%'; // View database character set
  mysql>SHOW VARIABLES LIKE 'collation_%';

  SET NAMES 'utf8';
  ```

+ Configuration `mysqld.cnf`

  ```
  # install vim
  apt-get update
  apt-get install vim

  vi /etc/mysql/mysql.conf.d/mysqld.cnf

  [mysqld]
  character_set_server = utf8

  [mysql]
  default-character-set = utf8

  [mysql.server]
  default-character-set = utf8

  [mysqld_safe]
  default-character-set = utf8

  [client]
  default-character-set = utf8
  ```

### Create Submarine Database

#### Create development database
Copy the files, submarine.sql, submarine-data.sql and metastore.sql to the mysql docker.

```
docker cp ${SUBMARINE_HOME}/docs/database/submarine.sql ${DOCKER_ID}:/
docker cp ${SUBMARINE_HOME}/docs/database/submarine-data.sql ${DOCKER_ID}:/
docker cp ${SUBMARINE_HOME}/docs/database/metastore.sql ${DOCKER_ID}:/
```

Development database for development environment.

```
# in mysql container
bash > mysql -uroot -ppassword
mysql> CREATE USER IF NOT EXISTS 'submarine'@'%' IDENTIFIED BY 'password';
mysql> GRANT ALL PRIVILEGES ON * . * TO 'submarine'@'%';
mysql> CREATE DATABASE IF NOT EXISTS submarine CHARACTER SET utf8 COLLATE utf8_general_ci;
mysql> use submarine;
mysql> source /submarine.sql;
mysql> source /submarine-data.sql;
mysql> CREATE USER IF NOT EXISTS 'metastore'@'%' IDENTIFIED BY 'password';
mysql> GRANT ALL PRIVILEGES ON * . * TO 'metastore'@'%';
mysql> CREATE DATABASE IF NOT EXISTS metastore CHARACTER SET utf8 COLLATE utf8_general_ci;
mysql> use metastore;
mysql> source /metastore.sql;
mysql> quit
```

>  NOTE: submarine development database name is  `submarine` and user name is `submarine`, password is `password`, metastore development database name is  `metastore` and user name is `metastore`, password is `password`, This is the default value in the system's `submarine-site.xml` configuration file and is not recommended for modification.


#### Create test database

Test database for program unit testing and Travis test environment.

```
# in mysql container
bash > mysql -uroot -ppassword
mysql> CREATE USER IF NOT EXISTS 'submarine_test'@'%' IDENTIFIED BY 'password_test';
mysql> GRANT ALL PRIVILEGES ON * . * TO 'submarine_test'@'%';
mysql> CREATE DATABASE IF NOT EXISTS `submarine_test` CHARACTER SET utf8 COLLATE utf8_general_ci;
mysql> use `submarine_test`;
mysql> source /submarine.sql;
mysql> CREATE USER IF NOT EXISTS 'metastore_test'@'%' IDENTIFIED BY 'password_test';
mysql> GRANT ALL PRIVILEGES ON * . * TO 'metastore_test'@'%';
mysql> CREATE DATABASE IF NOT EXISTS `metastore_test` CHARACTER SET utf8 COLLATE utf8_general_ci;
mysql> use `metastore_test`;
mysql> source /metastore.sql;
mysql> quit
```

>  NOTE: submarine test database name is  `submarine_test` and user name is `submarine_test`, password is `password_test`, metastore test database name is  `metastore_test` and user name is `metastore_test`, password is `password_test`, Cannot be configured, values that cannot be modified.

#### mysqldump

```$xslt
mysqldump -uroot -ppassword --databases submarine > submarine.sql;
mysqldump -umetastore -ppassword metastore > metastore.sql;
```


## Travis

1. In the submarine's Travis, the `test database`, `database name`, `username` and `password` will be automatically created based on the contents of this document.

   Therefore, do not modify the database's `database name`, `username` and `password` configuration to avoid introducing some problems.

2. In the mysql database in Travis, the `submarine.sql` are executed to create the submarine database table structure and test data.

3. The submarine database test case written in the `workbench-server` module will also be unit tested in the mysql database in travis.
