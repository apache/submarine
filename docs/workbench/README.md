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
# Submarine Workbench

Submarine Workbench is an UI designed for data scientists. 
Data scientists can interact with Submarine Workbench UI to access notebooks, 
submit/manage jobs, manage models, create model training workflows, access dataset, etc.

## Run Workbench

```$xslt
cd submarine
./bin/workbench-daemon.sh [start|stop|restart]
```
To start workbench server, you need to download mysql jdbc jar and put it in the
path of workbench/lib for the first time. Or you can add parameter, getMysqlJar,
to get mysql jar automatically.
```$xslt
cd submarine
./bin/workbench-daemon.sh start getMysqlJar
```

## submarine-env.sh

`submarine-env.sh` is automatically executed each time the `workbench.sh` script is executed, so we can set the `workbench.sh` script and the environment variables in the `WorkbenchServer` process via `submarine-env.sh`.

| Name                | Variable                                                     |
| ------------------- | ------------------------------------------------------------ |
| JAVA_HOME           | Set your java home path, default is `java`.                  |
| WORKBENCH_JAVA_OPTS | Set the JAVA OPTS parameter when the Workbench process starts. If you need to debug the Workbench process, you can set it to `-agentlib:jdwp=transport=dt_socket, server=y,suspend=n,address=5005` |
| WORKBENCH_MEM       | Set the java memory parameter when the Workbench process starts. |
| MYSQL_JAR_URL       | The customized URL to download mysql jdbc jar.               |
| MYSQL_VERSION       | The version of mysql jdbc jar would be downloaded. The default value is 5.1.39. It's used to generate the default value of MYSQL_JDBC_URL|

## submarine-site.xml

`submarine-site.xml` is the configuration file for the entire `Submarine` system to run.

| Name                               | Variable                                                     |
| ---------------------------------- | ------------------------------------------------------------ |
| workbench.server.addr              | workbench server address, default is `0.0.0.0`               |
| workbench.server.port              | workbench server port, default `8080`                        |
| workbench.ssl                      | Should SSL be used by the workbench servers?, default `false` |
| workbench.server.ssl.port          | Server ssl port. (used when ssl property is set to true), default `8483` |
| workbench.ssl.client.auth          | Should client authentication be used for SSL connections?    |
| workbench.ssl.keystore.path        | Path to keystore relative to submarine configuration directory |
| workbench.ssl.keystore.type        | The format of the given keystore (e.g. JKS or PKCS12)        |
| workbench.ssl.keystore.password    | Keystore password. Can be obfuscated by the Jetty Password tool |
| workbench.ssl.key.manager.password | Key Manager password. Defaults to keystore password. Can be obfuscated. |
| workbench.ssl.truststore.path      | Path to truststore relative to submarine configuration directory. Defaults to the keystore path |
| workbench.ssl.truststore.type      | The format of the given truststore (e.g. JKS or PKCS12). Defaults to the same type as the keystore type |
| workbench.ssl.truststore.password  | Truststore password. Can be obfuscated by the Jetty Password tool. Defaults to the keystore password |
| workbench.web.war                  | Submarine workbench web war file path.                       |



## Compile

[Build From Code Guide](./development/BuildFromCode.md)

```$xslt
cd submarine/submarine-dist/target/submarine-dist-<version>/submarine-dist-<version>/
./bin/workbench-daemon.sh [start|stop|restart]
```
