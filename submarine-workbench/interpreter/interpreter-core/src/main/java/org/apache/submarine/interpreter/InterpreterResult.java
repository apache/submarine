/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.interpreter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedList;
import java.util.List;

/**
 * Interpreter result template.
 */
public class InterpreterResult {
  private static final Logger LOG = LoggerFactory.getLogger(InterpreterResult.class);

  private Code code;
  private List<InterpreterResultMessage> msg = new LinkedList<>();

  public InterpreterResult(org.apache.zeppelin.interpreter.InterpreterResult result) {
    if (result.code() == org.apache.zeppelin.interpreter.InterpreterResult.Code.SUCCESS) {
      code = Code.SUCCESS;
    } else if (result.code() == org.apache.zeppelin.interpreter.InterpreterResult.Code.INCOMPLETE) {
      code = Code.INCOMPLETE;
    } else if (result.code() == org.apache.zeppelin.interpreter.InterpreterResult.Code.ERROR) {
      code = Code.ERROR;
    } else if (result.code() == org.apache.zeppelin.interpreter.InterpreterResult.Code.KEEP_PREVIOUS_RESULT) {
      code = Code.KEEP_PREVIOUS_RESULT;
    } else {
      LOG.error("Unknow code type : " + result.code());
    }

    for (org.apache.zeppelin.interpreter.InterpreterResultMessage message: result.message()) {
      add(message);
    }
  }

  public void add(org.apache.zeppelin.interpreter.InterpreterResultMessage message) {
    Type type = Type.TEXT;
    if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.TEXT) {
      type = Type.TEXT;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.HTML) {
      type = Type.HTML;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.ANGULAR) {
      type = Type.ANGULAR;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.TABLE) {
      type = Type.TABLE;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.IMG) {
      type = Type.IMG;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.SVG) {
      type = Type.SVG;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.NULL) {
      type = Type.NULL;
    } else if (message.getType() == org.apache.zeppelin.interpreter.InterpreterResult.Type.NETWORK) {
      type = Type.NETWORK;
    } else {
      LOG.error("Unknow type : " + message.getType());
    }

    InterpreterResultMessage interpreterResultMessage = new InterpreterResultMessage(type, message.getData());
    this.msg.add(interpreterResultMessage);
  }

  public Code code() {
    return this.code;
  }

  public List<InterpreterResultMessage> message() {
    return msg;
  }

  /**
   *  Type of result after code execution.
   */
  public enum Code {
    SUCCESS,
    INCOMPLETE,
    ERROR,
    KEEP_PREVIOUS_RESULT
  }

  /**
   * Type of Data.
   */
  public enum Type {
    TEXT,
    HTML,
    ANGULAR,
    TABLE,
    IMG,
    SVG,
    NULL,
    NETWORK
  }
}
