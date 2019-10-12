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
# Submarine Python Interpreter

## Test Submarine Python Interpreter

### Execute test command
```
java -jar python-interpreter-0.3.0-SNAPSHOT-shade.jar python python-interpreter-id test
```

### Print test result 
```
 INFO [2019-10-14 10:35:11,653] ({main} SubmarinePythonInterpreter.java[test]:111) - Execution Python Interpreter, Calculation formula 1 + 1, Result = 2
 INFO [2019-10-14 10:35:11,653] ({main} InterpreterProcess.java[main]:68) - Interpreter test result: true
```


## Debug Submarine Python Interpreter

### Execute debug command

```
java -jar python-interpreter-0.3.0-SNAPSHOT-shade.jar -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 python python-interpreter-id
```

Connect via remote debugging in IDEA
