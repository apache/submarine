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

# Building Submarine Spark Security Plugin

Submarine Spark Security Plugin is built using [Apache Maven](http://maven.apache.org). To build it, `cd` to the root direct of submarine project and run:

```bash
mvn clean package -Dmaven.javadoc.skip=true -DskipTests -pl :submarine-spark-security
```

By default, Submarine Spark Security Plugin is built against Apache Spark 2.3.x and Apache Ranger 1.1.0, which may be incompatible with other Apache Spark or Apache Ranger releases.

Currently, available profiles are:

Spark: -Pspark-2.3, -Pspark-2.4, -Pspark-3.0

Ranger: -Pranger-1.0, -Pranger-1.1, -Pranger-1.2 -Pranger-2.0
