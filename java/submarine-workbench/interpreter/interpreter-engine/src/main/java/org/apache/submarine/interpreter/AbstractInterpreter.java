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
import org.apache.zeppelin.interpreter.InterpreterResultMessage;
import org.apache.zeppelin.interpreter.InterpreterOutput;
import org.apache.zeppelin.interpreter.InterpreterGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Properties;

/**
 * Entry point for Submarine Interpreter process.
 * Accepting thrift connections from Submarine Workbench Server.
 */
public abstract class AbstractInterpreter implements Interpreter {
  private static final Logger LOG = LoggerFactory.getLogger(AbstractInterpreter.class);

  protected org.apache.zeppelin.interpreter.Interpreter zeppelinInterpreter;

  private InterpreterContext interpreterContext;

  protected InterpreterContext getIntpContext() {
    if (this.interpreterContext == null) {
      this.interpreterContext = InterpreterContext.builder()
              .setInterpreterOut(new InterpreterOutput(null))
              .build();
      InterpreterContext.set(this.interpreterContext);
    }
    return this.interpreterContext;
  }

  public void setInterpreterGroup(InterpreterGroup interpreterGroup) {
    this.zeppelinInterpreter.setInterpreterGroup(interpreterGroup);
  }

  public InterpreterGroup getInterpreterGroup() {
    return this.zeppelinInterpreter.getInterpreterGroup();
  }

  protected Properties mergeZeppelinInterpreterProperties(Properties properties) {
    Properties newProps = new Properties();

    for (String key : properties.stringPropertyNames()) {
      String newKey = key.replace("submarine", "zeppelin");
      newProps.put(newKey, properties.getProperty(key));
    }

    return newProps;
  }

  @Override
  public InterpreterResult interpret(String code) throws InterpreterException {
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
    } catch (org.apache.zeppelin.interpreter.InterpreterException | IOException e) {
      LOG.error(e.getMessage(), e);
      throw new InterpreterException(e);
    }

    return interpreterResult;
  }

  @Override
  public void open() throws InterpreterException {
    getIntpContext();
    try {
      this.zeppelinInterpreter.open();
    } catch (org.apache.zeppelin.interpreter.InterpreterException e) {
      LOG.error(e.getMessage(), e);
      throw new org.apache.submarine.interpreter.InterpreterException(e);
    }
  }

  @Override
  public void close() throws InterpreterException {
    try {
      this.zeppelinInterpreter.close();
    } catch (org.apache.zeppelin.interpreter.InterpreterException e) {
      LOG.error(e.getMessage(), e);
      throw new org.apache.submarine.interpreter.InterpreterException(e);
    }
  }

  @Override
  public void cancel() throws InterpreterException {
    try {
      this.zeppelinInterpreter.cancel(getIntpContext());
    } catch (org.apache.zeppelin.interpreter.InterpreterException e) {
      LOG.error(e.getMessage(), e);
      throw new org.apache.submarine.interpreter.InterpreterException(e);
    }
  }

  @Override
  public int getProgress() throws InterpreterException {
    int process = 0;
    try {
      process = this.zeppelinInterpreter.getProgress(getIntpContext());
    } catch (org.apache.zeppelin.interpreter.InterpreterException e) {
      LOG.error(e.getMessage(), e);
      throw new org.apache.submarine.interpreter.InterpreterException(e);
    }
    return process;
  }

  @Override
  public void addToSession(String session) {
    this.zeppelinInterpreter.getInterpreterGroup().get(session).add(this.zeppelinInterpreter);
  }
}
