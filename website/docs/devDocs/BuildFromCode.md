---
title: How to Build Submarine
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
## Prerequisites

+ JDK 1.8
+ Maven 3.3 or later ( < 3.8.1 )
+ Docker

## Quick Start

### Build Your Custom Submarine Docker Images

Submarine provides default Docker image in the release artifacts, sometimes you would like to do some modifications on the images. You can rebuild Docker image after you make changes.

> Note that you need to make sure the images built above can be accessed in k8s
> Usually this needs to rename and push to a proper Docker registry.

```bash
mvn clean package -DskipTests
```

Build submarine server image:

```bash
./dev-support/docker-images/submarine/build.sh
```

Build submarine database image:

```bash
./dev-support/docker-images/database/build.sh
```

### Building source code / binary distribution

+ Checking releases for licenses

```
mvn clean org.apache.rat:apache-rat-plugin:check
```

+ Create binary distribution with default hadoop version

```
mvn clean package -DskipTests
```

+ Create binary distribution with hadoop-2.9.x version

```
mvn clean package -DskipTests -Phadoop-2.9
```

+ Create binary distribution with hadoop-2.10.x version

```
mvn clean package -DskipTests -Phadoop-2.10
```

+ Create binary distribution with hadoop-3.1.x version

```
mvn clean package -DskipTests -Phadoop-3.1
```

+ Create binary distribution with hadoop-3.2.x version

```
mvn clean package -DskipTests -Phadoop-3.2
```

+ Create source code distribution

```
mvn clean package -DskipTests -Psrc
```

### Building source code / binary distribution with Maven Wrapper
+ Maven Wrapper (Optional): Maven Wrapper can help you avoid dependencies problem about Maven version.
```
# Setup Maven Wrapper (Maven 3.6.1)
mvn -N io.takari:maven:0.7.7:wrapper -Dmaven=3.6.1

# Check Maven Wrapper
./mvnw -version

# Replace 'mvn' with 'mvnw'. Example:
./mvnw clean package -DskipTests
```
