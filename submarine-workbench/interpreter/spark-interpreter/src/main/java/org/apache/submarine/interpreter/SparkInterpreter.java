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

import org.apache.commons.lang.StringUtils;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterException;
import org.apache.zeppelin.interpreter.InterpreterGroup;
import org.apache.zeppelin.interpreter.InterpreterResultMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class SparkInterpreter extends InterpreterProcess {
  private static final Logger LOG = LoggerFactory.getLogger(SparkInterpreter.class);

  private org.apache.zeppelin.spark.SparkInterpreter zpleSparkInterpreter;
  private InterpreterContext intpContext;

  private String extractScalaVersion() throws InterpreterException {
    String scalaVersionString = scala.util.Properties.versionString();
    LOG.info("Using Scala: " + scalaVersionString);
    if (scalaVersionString.contains("version 2.10")) {
      return "2.10";
    } else if (scalaVersionString.contains("version 2.11")) {
      return "2.11";
    } else if (scalaVersionString.contains("version 2.12")) {
      return "2.12";
    } else {
      throw new InterpreterException("Unsupported scala version: " + scalaVersionString);
    }
  }

  public SparkInterpreter(Properties properties) {
    properties = mergeZeplSparkIntpProp(properties);
    zpleSparkInterpreter = new org.apache.zeppelin.spark.SparkInterpreter(properties);
    zpleSparkInterpreter.setInterpreterGroup(new InterpreterGroup());
    intpContext = this.getIntpContext();
  }
  public SparkInterpreter() {
    this(new Properties());
  }

  @Override
  public void open() {
    try {
      ClassLoader scalaInterpreterClassLoader = null;
      String submarineHome = System.getenv("SUBMARINE_HOME");
      String interpreterDir  = "";
      if (StringUtils.isBlank(submarineHome)) {
        LOG.warn("SUBMARINE_HOME is not set, default interpreter directory is ../ ");
        interpreterDir = "..";
      } else {
        interpreterDir = submarineHome + "/workbench/interpreter";
      }
      String scalaVersion = extractScalaVersion();
      File scalaJarFolder = new File(interpreterDir + "/spark/scala-" + scalaVersion);
      List<URL> urls = new ArrayList<>();
      for (File file : scalaJarFolder.listFiles()) {
        LOG.info("Add file " + file.getAbsolutePath() + " to classpath of spark scala interpreter: "
                + scalaJarFolder);
        urls.add(file.toURI().toURL());
      }
      scalaInterpreterClassLoader = new URLClassLoader(urls.toArray(new URL[0]),
              Thread.currentThread().getContextClassLoader());
      if (scalaInterpreterClassLoader != null) {
        Thread.currentThread().setContextClassLoader(scalaInterpreterClassLoader);
      }
      zpleSparkInterpreter.open();
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    } catch (MalformedURLException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public InterpreterResult interpret(String code) {
    InterpreterResult interpreterResult = null;
    try {
      org.apache.zeppelin.interpreter.InterpreterResult zeplInterpreterResult
              = zpleSparkInterpreter.interpret(code, intpContext);
      interpreterResult = new InterpreterResult(zeplInterpreterResult);

      List<InterpreterResultMessage> interpreterResultMessages =
              intpContext.out.toInterpreterResultMessage();

      for (org.apache.zeppelin.interpreter.InterpreterResultMessage message : interpreterResultMessages) {
        interpreterResult.add(message);
      }
    } catch (InterpreterException | IOException e) {
      LOG.error(e.getMessage(), e);
    }

    return interpreterResult;
  }

  @Override
  public void close() {
    try {
      zpleSparkInterpreter.close();
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public void cancel() {
    try {
      zpleSparkInterpreter.cancel(intpContext);
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public int getProgress() {
    int process = 0;
    try {
      process = zpleSparkInterpreter.getProgress(intpContext);
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }

    return process;
  }

  @Override
  public boolean test() {
    open();
    String code = "val df = spark.createDataFrame(Seq((1,\"a\"),(2, null)))\n" +
        "df.show()";
    InterpreterResult result = interpret(code);
    LOG.info("Execution Spark Interpreter, Calculation Spark Code  {}, Result = {}",
             code, result.message().get(0).getData());

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
                "+---+----+"));
    close();
    return success;
  }
}
