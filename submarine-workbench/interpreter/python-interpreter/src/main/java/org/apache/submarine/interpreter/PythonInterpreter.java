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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Properties;

public class PythonInterpreter extends InterpreterProcess {
  private static final Logger LOG = LoggerFactory.getLogger(PythonInterpreter.class);

  private org.apache.zeppelin.python.PythonInterpreter zplePythonInterpreter;
  private InterpreterContext intpContext;

  public PythonInterpreter() {
    Properties properties = new Properties();
    properties = mergeZeplPyIntpProp(properties);
    zplePythonInterpreter = new org.apache.zeppelin.python.PythonInterpreter(properties);
    zplePythonInterpreter.setInterpreterGroup(new InterpreterGroup());
    intpContext = this.getIntpContext();
  }

  @Override
  public void open() {
    try {
      zplePythonInterpreter.open();
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public InterpreterResult interpret(String code) {
    InterpreterResult interpreterResult = null;
    try {
      org.apache.zeppelin.interpreter.InterpreterResult zeplInterpreterResult
          = zplePythonInterpreter.interpret(code, intpContext);
      interpreterResult = new InterpreterResult(zeplInterpreterResult);

      List<org.apache.zeppelin.interpreter.InterpreterResultMessage> interpreterResultMessages =
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
      zplePythonInterpreter.close();
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public void cancel() {
    try {
      zplePythonInterpreter.cancel(intpContext);
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  @Override
  public int getProgress() {
    int process = 0;
    try {
      process = zplePythonInterpreter.getProgress(intpContext);
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
    }

    return process;
  }

  @Override
  public boolean test() {
    open();
    String code = "1 + 1";
    InterpreterResult result = interpret(code);
    LOG.info("Execution Python Interpreter, Calculation formula {}, Result = {}",
        code, result.message().get(0).getData());

    if (result.code() != InterpreterResult.Code.SUCCESS) {
      return false;
    }
    if (StringUtils.equals(result.message().get(0).getData(), "2\n")) {
      return true;
    }
    return false;
  }
}
