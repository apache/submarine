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

import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterException;
import org.apache.zeppelin.interpreter.InterpreterResultMessage;
import org.apache.zeppelin.interpreter.InterpreterOutput;
import org.apache.zeppelin.interpreter.InterpreterGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

/**
 * Entry point for Submarine Interpreter process.
 * Accepting thrift connections from Submarine Workbench Server.
 */
public abstract class AbstractInterpreter implements Interpreter {
  private static final Logger LOG = LoggerFactory.getLogger(AbstractInterpreter.class);

  protected org.apache.zeppelin.interpreter.Interpreter zeppelinInterpreter;

  private InterpreterContext interpreterContext;

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

  public InterpreterContext getIntpContext() {
    if (this.interpreterContext == null) {
      this.interpreterContext = InterpreterContext.builder()
              .setInterpreterOut(new InterpreterOutput(null))
              .build();
      InterpreterContext.set(this.interpreterContext);
    }
    return this.interpreterContext;
  }

  public void setIntpContext(InterpreterContext context){
    InterpreterContext.set(context);
    this.interpreterContext = context;
  }

  public void setInterpreterGroup(InterpreterGroup interpreterGroup) {
    this.zeppelinInterpreter.setInterpreterGroup(interpreterGroup);
  }

  public InterpreterGroup getInterpreterGroup() {
    return this.zeppelinInterpreter.getInterpreterGroup();
  }

  @Override
  public InterpreterResult interpret(String code) {
    InterpreterResult interpreterResult = null;
    try {
      org.apache.zeppelin.interpreter.InterpreterResult zeplInterpreterResult
              = this.zeppelinInterpreter.interpret(code, getIntpContext());
      interpreterResult = new InterpreterResult(zeplInterpreterResult);

      List<InterpreterResultMessage> interpreterResultMessages =
              getIntpContext().out.toInterpreterResultMessage();

      for (org.apache.zeppelin.interpreter.InterpreterResultMessage message : interpreterResultMessages) {
        interpreterResult.add(message);
      }
    } catch (InterpreterException | IOException e) {
      LOG.error(e.getMessage(), e);
    }

    return interpreterResult;
  }

  @Override
  public void open() throws InterpreterException {
    getIntpContext();
    this.zeppelinInterpreter.open();
  }

  @Override
  public void close() throws InterpreterException {
    this.zeppelinInterpreter.close();
  }

  @Override
  public void cancel() {
    try {
      this.zeppelinInterpreter.cancel(getIntpContext());
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public int getProgress() {
    int process = 0;
    try {
      process = this.zeppelinInterpreter.getProgress(getIntpContext());
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
    return process;
  }

  @Override
  public void addToSession(String session) {
    this.zeppelinInterpreter.getInterpreterGroup().get(session).add(this.zeppelinInterpreter);
  }
}
