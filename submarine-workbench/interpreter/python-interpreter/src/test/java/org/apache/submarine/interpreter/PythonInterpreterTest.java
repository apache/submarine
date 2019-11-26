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

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class PythonInterpreterTest {
  private static final Logger LOG = LoggerFactory.getLogger(PythonInterpreterTest.class);

  private static PythonInterpreter pythonInterpreterForCancel = null;

  private static PythonInterpreter pythonInterpreterForClose = null;

  @BeforeClass
  public static void setUp() throws InterpreterException {
    pythonInterpreterForCancel = new PythonInterpreter();
    pythonInterpreterForClose = new PythonInterpreter();
    pythonInterpreterForCancel.open();
    pythonInterpreterForClose.open();
  }

  @AfterClass
  public static void tearDown() throws InterpreterException {
    if (null != pythonInterpreterForCancel) {
      pythonInterpreterForCancel.close();
    }
    if (null != pythonInterpreterForClose) {
      pythonInterpreterForClose.close();
    }
  }


  @Test
  public void calcOnePlusOne() throws InterpreterException {
    String code = "1+1";
    InterpreterResult result = pythonInterpreterForCancel.interpret(code);
    LOG.info("result = {}", result);

    assertEquals(result.code(), InterpreterResult.Code.SUCCESS);
    // 1 + 1 = 2 + '\n'
    assertEquals(result.message().get(0).getData(), "2\n");
  }

  private static class infinityPythonJobforCancel implements Runnable {
    @Override
    public void run() {
      String code = "import time\nwhile True:\n  time.sleep(1)";
      InterpreterResult ret = null;
      try {
        ret = pythonInterpreterForCancel.interpret(code);
        assertNotNull(ret);
        Pattern expectedMessage = Pattern.compile("KeyboardInterrupt");
        Matcher m = expectedMessage.matcher(ret.message().toString());
        assertTrue(m.find());
      } catch (InterpreterException e) {
        e.printStackTrace();
        fail();
      }
    }
  }

  private static class infinityPythonJobforClose implements Runnable {
    @Override
    public void run() {
      String code = "import time\nwhile True:\n  time.sleep(1)";
      try {
        pythonInterpreterForClose.interpret(code);
      } catch (org.apache.submarine.interpreter.InterpreterException e) {
        e.printStackTrace();
        fail();
      }
    }
  }


  @Test
  public void testCloseIntp() throws InterruptedException, InterpreterException {
    assertEquals(InterpreterResult.Code.SUCCESS,
            pythonInterpreterForClose.interpret("1+1\n").code());
    Thread t = new Thread(new infinityPythonJobforClose());
    t.start();
    Thread.sleep(5000);
    pythonInterpreterForClose.close();
    assertTrue(t.isAlive());
    t.join(2000);
    assertFalse(t.isAlive());
  }


  @Test
  public void testCancelIntp() throws InterruptedException, InterpreterException {
    assertEquals(InterpreterResult.Code.SUCCESS,
            pythonInterpreterForCancel.interpret("1+1\n").code());
    Thread t = new Thread(new infinityPythonJobforCancel());
    t.start();
    Thread.sleep(5000);
    pythonInterpreterForCancel.cancel();
    assertTrue(t.isAlive());
    t.join(3000);
    assertFalse(t.isAlive());
  }


}
