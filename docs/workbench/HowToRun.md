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
# How To Run Submarine Workbench
We provide two methods to launch Submarine Workbench
*  Method 1:  Run Submarine Workbench on docker
*  Method 2:  Run Submarine Workbench without docker

# Run Submarine Workbench on docker

By using the official images of Submarine, only a few docker commands are required to launch **Submarine Workbench**. The document includes information about how to launch the Submarine Workbench via the new docker images and the information about how to switch between different Submarine Workbench versions(version Vue & version Angular).

### Two versions of Submarine Workbench
1. Angular (default)
2. Vue (This is the old version, and it will be replaced by version Angular in the future.)
#### (WARNING: Please restart a new **incognito window** when you switch to different versions of Submarine Workbench)
### Launch the Submarine Workbench(Angular)
* It should be noted that since Submarine Workbench depends on the Submarine database, so you need to run the docker container of the Submarine database first.
```
docker run -it -p 3306:3306 -d --name submarine-database -e MYSQL_ROOT_PASSWORD=password apache/submarine:database-0.4.0
docker run -it -p 8080:8080 -d --link=submarine-database:submarine-database --name submarine-server apache/submarine:server-0.4.0
```
* The login page of Submarine Workbench will be shown in ```http://127.0.0.1:8080```.

### Switch from version Angular to version Vue
*  Step1: Launch submarine-database and submarine-server containers
```
docker run -it -p 3306:3306 -d --name submarine-database -e MYSQL_ROOT_PASSWORD=password apache/submarine:database-0.4.0
docker run -it -p 8080:8080 -d --link=submarine-database:submarine-database --name submarine-server apache/submarine:server-0.4.0
```
*  Step2: Compile Submarine in your host (not in the container)
```
cd ./submarine
mvn clean install package -DskipTests
```
*  Step3: Copy workbench-web.war into the submarine-server container
```
cd submarine-workbench/workbench-web/target
docker cp workbench-web.war submarine-server:/opt/submarine-dist-0.4.0-hadoop-2.9
```
*  Step4: Enter the submarine-server container
```
docker exec -it submarine-server bash
```
*  Step5: Modify the value of the configuration **workbench.web.war** in conf/submarine-site.xml from "../submarine-workbench-web-ng.war" to "../submarine-workbench-web.war".

*  Step6: Restart the Submarine Server
```
./bin/submarine-daemon.sh restart
```
*  Step7: Launch the submarine-server container
```
docker start submarine-server
```
*  Step8: Open a new **incognito window(not a tab)** and check ```http://127.0.0.1:8080```
### Switch from version Vue to version Angular
*  Step1: Enter the submarine-server container
```
docker exec -it submarine-server bash
```
*  Step2: Modify the value of the configuration **workbench.web.war** in conf/submarine-site.xml from "../workbench-web.war" to "../submarine-workbench-web-ng.war".
*  Step3: Restart the Submarine Server
```
./bin/submarine-daemon.sh restart
```
*  Step4: Launch the submarine-server container
```
docker start submarine-server
```
*  Step5: Open a **new incognito window(not a tab)** and check ```http://127.0.0.1:8080```
### Check the data in the submarine-database
*  Step1: Enter the submarine-database container
```
docker exec -it submarine-database bash
```
*  Step2: Enter MySQL database
```
mysql -uroot -ppassword
```
*  Step3: List the data in the table
```
// list all databases
show databases;

// choose a database
use ${target_database};

// list all tables
show tables;

// list the data in the table
select * from ${target_table};
```
# Run Submarine Workbench without docker
### Run Submarine Workbench

```
cd submarine
./bin/submarine-daemon.sh [start|stop|restart]
```
To start workbench server, you need to download MySQL jdbc jar and put it in the
path of workbench/lib for the first time. Or you can add parameter, getMysqlJar,
to get MySQL jar automatically.
```
cd submarine
./bin/submarine-daemon.sh start getMysqlJar
```

### submarine-env.sh

`submarine-env.sh` is automatically executed each time the `submarine-daemon.sh` script is executed, so we can set the `submarine-daemon.sh` script and the environment variables in the `SubmarineServer` process via `submarine-env.sh`.

| Name                | Variable                                                     |
| ------------------- | ------------------------------------------------------------ |
| JAVA_HOME           | Set your java home path, default is `java`.                  |
| SUBMARINE_JAVA_OPTS | Set the JAVA OPTS parameter when the Submarine Workbench process starts. If you need to debug the Submarine Workbench process, you can set it to `-agentlib:jdwp=transport=dt_socket, server=y,suspend=n,address=5005` |
| SUBMARINE_MEM       | Set the java memory parameter when the Submarine Workbench process starts. |
| MYSQL_JAR_URL       | The customized URL to download MySQL jdbc jar.               |
| MYSQL_VERSION       | The version of MySQL jdbc jar to downloaded. The default value is 5.1.39. It's used to generate the default value of MYSQL_JDBC_URL |

### submarine-site.xml

`submarine-site.xml` is the configuration file for the entire `Submarine` system to run.

| Name                               | Variable                                                     |
| ---------------------------------- | ------------------------------------------------------------ |
| submarine.server.addr              | Submarine server address, default is `0.0.0.0`               |
| submarine.server.port              | Submarine server port, default `8080`                        |
| submarine.ssl                      | Should SSL be used by the Submarine servers?, default `false` |
| submarine.server.ssl.port          | Server ssl port. (used when ssl property is set to true), default `8483` |
| submarine.ssl.client.auth          | Should client authentication be used for SSL connections?    |
| submarine.ssl.keystore.path        | Path to keystore relative to Submarine configuration directory |
| submarine.ssl.keystore.type        | The format of the given keystore (e.g. JKS or PKCS12)        |
| submarine.ssl.keystore.password    | Keystore password. Can be obfuscated by the Jetty Password tool |
| submarine.ssl.key.manager.password | Key Manager password. Defaults to keystore password. Can be obfuscated. |
| submarine.ssl.truststore.path      | Path to truststore relative to Submarine configuration directory. Defaults to the keystore path |
| submarine.ssl.truststore.type      | The format of the given truststore (e.g. JKS or PKCS12). Defaults to the same type as the keystore type |
| submarine.ssl.truststore.password  | Truststore password. Can be obfuscated by the Jetty Password tool. Defaults to the keystore password |
| workbench.web.war                  | Submarine Workbench web war file path.                       |



### Compile

[Build From Code Guide](../development/BuildFromCode.md)

```$xslt
cd submarine/submarine-dist/target/submarine-dist-<version>/submarine-dist-<version>/
./bin/submarine-daemon.sh [start|stop|restart]
```
