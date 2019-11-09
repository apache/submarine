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
# Submarine Spark Interpreter

## Test Submarine Spark Interpreter

### Execute test command
```
export SUBMARINE_HOME=/path/to/your/submarine_home
java -jar spark-interpreter-0.3.0-SNAPSHOT-shade.jar spark spark-interpreter-id test
```

### Print test result 
```
 INFO [2019-11-09 11:12:04,888] ({main} ContextHandler.java[doStart]:781) - Started o.s.j.s.ServletContextHandler@58b97c15{/stages/stage/kill,null,AVAILABLE,@Spark}
 INFO [2019-11-09 11:12:04,889] ({main} Logging.scala[logInfo]:54) - Bound SparkUI to 0.0.0.0, and started at http://10.0.0.3:4040
 INFO [2019-11-09 11:12:04,923] ({main} Logging.scala[logInfo]:54) - Starting executor ID driver on host localhost
 INFO [2019-11-09 11:12:04,927] ({main} Logging.scala[logInfo]:54) - Using REPL class URI: spark://10.0.0.3:64837/classes
 INFO [2019-11-09 11:12:04,937] ({main} Logging.scala[logInfo]:54) - Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 64838.
 INFO [2019-11-09 11:12:04,937] ({main} Logging.scala[logInfo]:54) - Server created on 10.0.0.3:64838
 INFO [2019-11-09 11:12:04,938] ({main} Logging.scala[logInfo]:54) - Using org.apache.spark.storage.RandomBlockReplicationPolicy for block replication policy
 INFO [2019-11-09 11:12:04,950] ({main} Logging.scala[logInfo]:54) - Registering BlockManager BlockManagerId(driver, 10.0.0.3, 64838, None)
 INFO [2019-11-09 11:12:04,952] ({dispatcher-event-loop-10} Logging.scala[logInfo]:54) - Registering block manager 10.0.0.3:64838 with 2004.6 MB RAM, BlockManagerId(driver, 10.0.0.3, 64838, None)
 INFO [2019-11-09 11:12:04,954] ({main} Logging.scala[logInfo]:54) - Registered BlockManager BlockManagerId(driver, 10.0.0.3, 64838, None)
 INFO [2019-11-09 11:12:04,954] ({main} Logging.scala[logInfo]:54) - Initialized BlockManager: BlockManagerId(driver, 10.0.0.3, 64838, None)
 INFO [2019-11-09 11:12:07,727] ({main} ContextHandler.java[doStart]:781) - Started o.s.j.s.ServletContextHandler@7aac6d13{/SQL/execution/json,null,AVAILABLE,@Spark}
 INFO [2019-11-09 11:12:07,735] ({main} ContextHandler.java[doStart]:781) - Started o.s.j.s.ServletContextHandler@89017e5{/static/sql,null,AVAILABLE,@Spark}                                                                                                                                                                                                          +- LocalRelation <empty>, [_1#0, _2#1]
 INFO [2019-11-09 11:12:08,499] ({main} Logging.scala[logInfo]:54) - Code generated in 81.495131 ms
 INFO [2019-11-09 11:12:08,518] ({main} Logging.scala[logInfo]:54) - Code generated in 11.738758 ms
 INFO [2019-11-09 11:12:08,525] ({main} SparkInterpreter.java[test]:159) - Execution Spark Interpreter, Calculation Spark Code  val df = spark.createDataFrame(Seq((1,"a"),(2, null)))
 
 df.show(), Result = +---+----+
 | _1 |  _2|
 +---+----+
 |  1|   a|
 |  2|null|
 +---+----+

 df: org.apache.spark.sql.DataFrame = [_1: int, _2: string]
 INFO [2019-11-09 11:12:08,525] ({main} SparkInterpreter.java[close]:159) - Close SparkInterpreter
 INFO [2019-11-09 11:12:08,536] ({main} Logging.scala[logInfo]:54) - Stopped Spark web UI at http://10.0.0.3:4040
 INFO [2019-11-09 11:12:08,539] ({dispatcher-event-loop-15} Logging.scala[logInfo]:54) - MapOutputTrackerMasterEndpoint stopped!
 INFO [2019-11-09 11:12:08,542] ({main} Logging.scala[logInfo]:54) - MemoryStore cleared
 INFO [2019-11-09 11:12:08,542] ({main} Logging.scala[logInfo]:54) - BlockManager stopped
 INFO [2019-11-09 11:12:08,544] ({main} Logging.scala[logInfo]:54) - BlockManagerMaster stopped
 INFO [2019-11-09 11:12:08,546] ({dispatcher-event-loop-4} Logging.scala[logInfo]:54) - OutputCommitCoordinator stopped!
 INFO [2019-11-09 11:12:08,551] ({main} Logging.scala[logInfo]:54) - Successfully stopped SparkContext
 INFO [2019-11-09 11:12:08,551] ({main} Logging.scala[logInfo]:54) - SparkContext already stopped.
 INFO [2019-11-09 11:12:08,551] ({main} InterpreterProcess.java[<init>]:120) - Interpreter test result: true
 INFO [2019-11-09 11:12:08,552] ({shutdown-hook-0} Logging.scala[logInfo]:54) - Shutdown hook called
 INFO [2019-11-09 11:12:08,553] ({shutdown-hook-0} Logging.scala[logInfo]:54) - Deleting directory /private/var/folders/xl/_xb3fgzj5zd698khfz6z74cc0000gn/T/spark-2f4acad9-a72d-4bca-8d85-3ef310f0b08c
```


## Debug Submarine Spark Interpreter

### Execute debug command

```
java -jar -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 spark-interpreter-0.3.0-SNAPSHOT-shade.jar spark spark-interpreter-id
```

Connect via remote debugging in IDEA
