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

## Introduction to MyBatis Generator Usage

## summary 

[Official Website](http://mybatis.org/generator/ "link")

MyBatis Generator (MBG) is a code generator for MyBatis MyBatis and iBATIS. 
It will generate code for all versions of MyBatis, and versions of iBATIS after 
version 2.2.0. It will introspect a database table (or many tables) and will 
generate artifacts that can be used to access the table(s). This lessens the 
initial nuisance of setting up objects and configuration files to interact 
with database tables. MBG seeks to make a major impact on the large percentage 
of database operations that are simple CRUD (Create, Retrieve, Update, Delete). 
You will still need to hand code SQL and objects for join queries, or stored procedures.

## Quick Start

### Add plug-in dependencies in pom.xml
The plug-in has been added in the pom.xml of the _submarine-server_.

```
<dependency>
  <groupId>org.mybatis.generator</groupId>
  <artifactId>mybatis-generator-core</artifactId>
  <version>1.3.7</version>
</dependency>
```

### Add plug-in dependencies in pom.xml
Edit the mbgConfiguration.xml file. We need to modify the following: 
1. We need to modify the JDBC connection information, such as driverClass, 
connectionURL, userId, password.
2. targetProject: You can specify a specific path as file storage path. e.g./tmp.
3. **tableName** and **domainObjectName**: List all the table to generate the code.

### Add main class
We have been added main class named _MybatisGeneratorMain_ in the _submarine-server_ 
project _org.apache.submarine.database.utils_ package path.

### Generator file
Run the main method to get the file, We can find the file under the 
targetProject path that we just configured. including: entity, TableNameMapper.java,
TableNameMapper.xml
