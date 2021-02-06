---
title: Submarine Spark Security Plugin
---
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

ACL Management for Apache Spark SQL with Apache Ranger, enabling:

- Table/Column level authorization
- Row level filtering
- Data masking


Security is one of fundamental features for enterprise adoption. [Apache Ranger™](https://ranger.apache.org) offers many security plugins for many Hadoop ecosystem components, 
such as HDFS, Hive, HBase, Solr and Sqoop2. However, [Apache Spark™](http://spark.apache.org) is not counted in yet. 
When a secured HDFS cluster is used as a data warehouse accessed by various users and groups via different applications wrote by Spark and Hive, 
it is very difficult to guarantee data management in a consistent way.  Apache Spark users visit data warehouse only 
with Storage based access controls offered by HDFS. This library enables Spark with SQL Standard Based Authorization. 

## Build

Please refer to the online documentation - [Building submarine spark security plguin](build-submarine-spark-security-plugin.md)

## Quick Start

Three steps to integrate Apache Spark and Apache Ranger.

### Installation

Place the submarine-spark-security-&lt;version&gt;.jar into `$SPARK_HOME/jars`.

### Configurations

#### Settings for Apache Ranger

Create `ranger-spark-security.xml` in `$SPARK_HOME/conf` and add the following configurations
for pointing to the right Apache Ranger admin server.


```xml

<configuration>

    <property>
        <name>ranger.plugin.spark.policy.rest.url</name>
        <value>ranger admin address like http://ranger-admin.org:6080</value>
    </property>

    <property>
        <name>ranger.plugin.spark.service.name</name>
        <value>a ranger hive service name</value>
    </property>

    <property>
        <name>ranger.plugin.spark.policy.cache.dir</name>
        <value>./a ranger hive service name/policycache</value>
    </property>

    <property>
        <name>ranger.plugin.spark.policy.pollIntervalMs</name>
        <value>5000</value>
    </property>

    <property>
        <name>ranger.plugin.spark.policy.source.impl</name>
        <value>org.apache.ranger.admin.client.RangerAdminRESTClient</value>
    </property>

</configuration>
```

Create `ranger-spark-audit.xml` in `$SPARK_HOME/conf` and add the following configurations
to enable/disable auditing.

```xml
<configuration>

    <property>
        <name>xasecure.audit.is.enabled</name>
        <value>true</value>
    </property>

    <property>
        <name>xasecure.audit.destination.db</name>
        <value>false</value>
    </property>

    <property>
        <name>xasecure.audit.destination.db.jdbc.driver</name>
        <value>com.mysql.jdbc.Driver</value>
    </property>

    <property>
        <name>xasecure.audit.destination.db.jdbc.url</name>
        <value>jdbc:mysql://10.171.161.78/ranger</value>
    </property>

    <property>
        <name>xasecure.audit.destination.db.password</name>
        <value>rangeradmin</value>
    </property>

    <property>
        <name>xasecure.audit.destination.db.user</name>
        <value>rangeradmin</value>
    </property>

</configuration>

```

#### Settings for Apache Spark

You can configure `spark.sql.extensions` with the `*Extension` we provided.
For example, `spark.sql.extensions=org.apache.submarine.spark.security.api.RangerSparkAuthzExtension`

Currently, you can set the following options to `spark.sql.extensions` to choose authorization w/ or w/o
extra functions.

| option | authorization | row filtering | data masking |
|---|---|---|---|
|org.apache.submarine.spark.security.api.RangerSparkAuthzExtension| √ | × | × |
|org.apache.submarine.spark.security.api.RangerSparkSQLExtension| √ | √ | √ |
