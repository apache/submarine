/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.submarine.interpreter;

import org.apache.zeppelin.interpreter.InterpreterException;
import org.apache.zeppelin.interpreter.InterpreterGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedList;
import java.util.Properties;

public class SparkSqlInterpreter extends AbstractInterpreter {
  private static final Logger LOG = LoggerFactory.getLogger(SparkInterpreter.class);

  public SparkSqlInterpreter(Properties properties) {
    properties = SparkInterpreter.mergeZeplSparkIntpProp(properties);
    this.zeppelinInterpreter = new org.apache.zeppelin.spark.SparkSqlInterpreter(properties);
  }

  public SparkSqlInterpreter() {
    this(new Properties());
  }


  @Override
  public boolean test() {
    try {
      InterpreterGroup intpGroup = new InterpreterGroup();
      SparkInterpreter sparkInterpreter = new SparkInterpreter();
      sparkInterpreter.setInterpreterGroup(intpGroup);

      this.setInterpreterGroup(intpGroup);

      String session = "session_1";
      intpGroup.put(session, new LinkedList<>());
      sparkInterpreter.addToSession(session);
      this.addToSession(session);

      sparkInterpreter.open();
      open();

      sparkInterpreter.interpret("case class Person(name:String, age:Int)");
      sparkInterpreter.interpret("case class People(group:String, person:Person)");
      sparkInterpreter.interpret(
              "val gr = sc.parallelize(Seq(" +
                      "People(\"g1\", " +
                      "Person(\"moon\",33)), " +
                      "People(\"g2\", " +
                      "Person(\"sun\",11" +
                      "))))");
      sparkInterpreter.interpret("gr.toDF.registerTempTable(\"gr\")");

      InterpreterResult result = interpret("select * from gr");
      LOG.info("Execution SparkSQL Interpreter, Calculation Spark Code {}, Result =\n {}",
               result.code(), result.message().get(0).getData()
      );
      if (result.code() != InterpreterResult.Code.SUCCESS) {
        return false;
      }
      return true;
    } catch (InterpreterException e) {
      e.printStackTrace();
      return false;
    }
  }
}
