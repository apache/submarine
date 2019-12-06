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
import org.apache.zeppelin.interpreter.InterpreterException;
import org.apache.zeppelin.interpreter.InterpreterGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class PythonInterpreter extends AbstractInterpreter {
  private static final Logger LOG = LoggerFactory.getLogger(PythonInterpreter.class);

  public PythonInterpreter() {
    this(new Properties());
  }

  public PythonInterpreter(Properties properties){
    properties = mergeZeppelinInterpreterProperties(properties);
    this.zeppelinInterpreter = new org.apache.zeppelin.python.PythonInterpreter(properties);
    this.zeppelinInterpreter.setInterpreterGroup(new InterpreterGroup());
  }

  @Override
  protected Properties mergeZeppelinInterpreterProperties(Properties properties) {
    properties = super.mergeZeppelinInterpreterProperties(properties);
    Properties newProps = new Properties();
    // Max number of dataframe rows to display.
    newProps.setProperty("zeppelin.python.maxResult", "1000");
    // whether use IPython when it is available
    newProps.setProperty("zeppelin.python.useIPython", "false");
    newProps.setProperty("zeppelin.python.gatewayserver_address", "127.0.0.1");

    if (null != properties) {
      newProps.putAll(properties);
    }
    return newProps;
  }

  @Override
  public boolean test() {
    try {
      open();
      String code = "1 + 1";
      InterpreterResult result = interpret(code);
      LOG.info("Execution Python Interpreter, Calculation formula {}, Result = {}",
               code, result.message().get(0).getData()
      );
      if (result.code() != InterpreterResult.Code.SUCCESS) {
        return false;
      }
      if (StringUtils.equals(result.message().get(0).getData(), "2\n")) {
        return true;
      }
      return false;
    } catch (InterpreterException e) {
      LOG.error(e.getMessage(), e);
      return false;
    }
  }
}
