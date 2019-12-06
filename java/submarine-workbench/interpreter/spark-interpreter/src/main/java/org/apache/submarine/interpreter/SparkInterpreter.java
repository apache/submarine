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

import org.apache.spark.SparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class SparkInterpreter extends AbstractInterpreter {
  private static final Logger LOG = LoggerFactory.getLogger(SparkInterpreter.class);

  public SparkInterpreter(Properties properties) {
    properties = mergeZeppelinInterpreterProperties(properties);
    this.zeppelinInterpreter = new org.apache.zeppelin.spark.SparkInterpreter(properties);
    this.setInterpreterGroup(new InterpreterGroup());
  }

  public SparkInterpreter() {
    this(new Properties());
  }

  @Override
  public boolean test() {
    try {
      open();
      String code = "val df = spark.createDataFrame(Seq((1,\"a\"),(2, null)))\n" +
              "df.show()";
      InterpreterResult result = interpret(code);
      LOG.info("Execution Spark Interpreter, Calculation Spark Code  {}, Result = {}",
               code, result.message().get(0).getData()
      );

      if (result.code() != InterpreterResult.Code.SUCCESS) {
        close();
        return false;
      }
      boolean success = (result.message().get(0).getData().contains(
              "+---+----+\n" +
              "| _1|  _2|\n" +
              "+---+----+\n" +
              "|  1|   a|\n" +
              "|  2|null|\n" +
              "+---+----+")
      );
      close();
      return success;
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
      return false;
    }
  }

  public SparkContext getSparkContext() {
    return ((org.apache.zeppelin.spark.SparkInterpreter) this.zeppelinInterpreter).getSparkContext();
  }

  public void setSchedulerPool(String pool){
    this.getIntpContext().getLocalProperties().put("pool", pool);
  }

  @Override
  protected Properties mergeZeppelinInterpreterProperties(Properties properties) {
    properties = super.mergeZeppelinInterpreterProperties(properties);
    Properties defaultProperties = new Properties();

    defaultProperties.setProperty("zeppelin.spark.maxResult", "1000");
    defaultProperties.setProperty("zeppelin.spark.scala.color", "false");

    if (null != properties) {
      defaultProperties.putAll(properties);
    }
    return defaultProperties;
  }

}
