<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Update Submarine Version

You can follow the documentation instructions.

Update your own modified and compiled submarine package to the submarine container, to testing.

### Build Submarine

```
cd submarine-project-dir/
mvn clean install package -DskipTests
```

### Copy submarine jar package to submarine container

```
docker cp submarine-all/target/submarine-all-<SUBMARINE_VERSION>-hadoop-<HADOOP_VERSION>.jar <container-id>:/tmp/
```

### Modify environment variables

```
cd /home/yarn/submarine
vi run_submarine-all_minist.sh

# Need to modify environment variables based on hadoop and submarine version numbers
SUBMARINE_VERSION=<submarine-version-number>
HADOOP_VERSION=<hadoop-version-number> # default 2.9
```

### Test submarine jar package in container

```
cd /home/yarn/submarine
./run_submarine-all_minist.sh
```

## Question and answer

1. Submarine package name error

   Because the package name of submarine 0.3.0 or higher has been changed from `apache.hadoop.submarine` to `apache.submarine`, So you need to set the Runtime settings in the `/usr/local/hadoop/etc/hadoop/submarine.xml` file.

   ```
   <configuration>
      <property>
        <name>submarine.runtime.class</name>
        <!--
        <value>org.apache.hadoop.yarn.submarine.runtimes.tony.TonyRuntimeFactory</value>
        -->
        <value>org.apache.submarine.runtimes.tony.TonyRuntimeFactory</value>
      </property>
   </configuration>
   ```
