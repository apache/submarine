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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class SparkInterpreterTest {
  private SparkInterpreter interpreter;

  // catch the streaming output in onAppend
  private volatile String output = "";

  @Before
  public void setUp() {
    // SparkInterpreter will change the current thread's classLoader
    Thread.currentThread().setContextClassLoader(SparkInterpreterTest.class.getClassLoader());
  }

  @Test
  public void testSparkInterpreter() throws InterruptedException, InterpreterException {
    Properties properties = new Properties();
    properties.setProperty("spark.master", "local");
    properties.setProperty("spark.app.name", "test");
    properties.setProperty("submarine.spark.scala.color", "false");
    properties.setProperty("submarine.spark.test", "true");
    properties.setProperty("submarine.spark.uiWebUrl", "fake_spark_weburl");
    // disable color output for easy testing
    properties.setProperty("submarine.spark.deprecatedMsg.show", "false");

    interpreter = new SparkInterpreter(properties);

    interpreter.open();

    InterpreterResult result = interpreter.interpret("val a=\"hello world\"");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertEquals("a: String = hello world\n", result.message().get(0).getData());

    result = interpreter.interpret("print(a)");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertEquals("hello world", result.message().get(0).getData());

    // java stdout
    result = interpreter.interpret("System.out.print(a)");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertEquals("hello world", result.message().get(0).getData());

    // incomplete
    result = interpreter.interpret("println(a");
    assertEquals(InterpreterResult.Code.INCOMPLETE, result.code());

    // syntax error
    result = interpreter.interpret("println(b)");
    assertEquals(InterpreterResult.Code.ERROR, result.code());
    assertTrue(result.message().get(0).getData().contains("not found: value b"));

    //multiple line
    result = interpreter.interpret("\"123\".\ntoInt");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // single line comment
    result = interpreter.interpret("print(\"hello world\")/*comment here*/");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertEquals("hello world", result.message().get(0).getData());

    result = interpreter.interpret("/*comment here*/\nprint(\"hello world\")");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // multiple line comment
    result = interpreter.interpret("/*line 1 \n line 2*/");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // test function
    result = interpreter.interpret("def add(x:Int, y:Int)\n{ return x+y }");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    result = interpreter.interpret("print(add(1,2))");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    result = interpreter.interpret("/*line 1 \n line 2*/print(\"hello world\")");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // Companion object with case class
    result = interpreter.interpret(
            "import scala.math._\n" +
                  "object Circle {\n" +
                  "private def calculateArea(radius: Double): Double = Pi * pow(radius, 2.0)\n" +
                  "}\n" +
                  "case class Circle(radius: Double) {\n" +
                  "  import Circle._\n" +
                  "  def area: Double = calculateArea(radius)\n" +
                  "}\n" +
                  "\n" +
                  "val circle1 = new Circle(5.0)");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // class extend
    result = interpreter.interpret("import java.util.ArrayList");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    result = interpreter.interpret("sc\n.range(1, 10)\n.sum");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertTrue(result.message().get(0).getData().contains("45"));
    result = interpreter.interpret("sc\n.range(1, 10)\n.sum");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertTrue(result.message().get(0).getData().contains("45"));
    result = interpreter.interpret("val bankText = sc.textFile(\"bank.csv\")");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    result = interpreter.interpret(
            "case class Bank(age:Integer, job:String, marital : String, edu : String, balance : Integer)\n" +
                  "val bank = bankText.map(s=>s.split(\";\")).filter(s => s(0)!=\"\\\"age\\\"\").map(\n" +
                  "    s => Bank(s(0).toInt, \n" +
                  "            s(1).replaceAll(\"\\\"\", \"\"),\n" +
                  "            s(2).replaceAll(\"\\\"\", \"\"),\n" +
                  "            s(3).replaceAll(\"\\\"\", \"\"),\n" +
                  "            s(5).replaceAll(\"\\\"\", \"\").toInt\n" +
                  "        )\n" +
                  ").toDF()");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // spark version
    result = interpreter.interpret("sc.version");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // spark sql test
    String version = result.message().get(0).getData().trim();
    if (version.contains("String = 1.")) {
      result = interpreter.interpret("sqlContext");
      assertEquals(InterpreterResult.Code.SUCCESS, result.code());

      result = interpreter.interpret(
              "val df = sqlContext.createDataFrame(Seq((1,\"a\"),(2, null)))\n" +
                    "df.show()");
      assertEquals(InterpreterResult.Code.SUCCESS, result.code());
      assertTrue(result.message().get(0).getData().contains(
              "+---+----+\n" +
              "| _1|  _2|\n" +
              "+---+----+\n" +
              "|  1|   a|\n" +
              "|  2|null|\n" +
              "+---+----+"));
    } else if (version.contains("String = 2.")) {
      result = interpreter.interpret("spark");
      assertEquals(InterpreterResult.Code.SUCCESS, result.code());

      result = interpreter.interpret(
              "val df = spark.createDataFrame(Seq((1,\"a\"),(2, null)))\n" +
                      "df.show()");
      assertEquals(InterpreterResult.Code.SUCCESS, result.code());
      assertTrue(result.message().get(0).getData().contains(
              "+---+----+\n" +
              "| _1|  _2|\n" +
              "+---+----+\n" +
              "|  1|   a|\n" +
              "|  2|null|\n" +
              "+---+----+"));
    }

    // submarineContext
    result = interpreter.interpret("z.show(df)");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertEquals(InterpreterResult.Type.TABLE, result.message().get(0).getType());
    assertEquals("_1\t_2\n1\ta\n2\tnull\n", result.message().get(0).getData());

    result = interpreter.interpret("z.input(\"name\", \"default_name\")");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

    // getProgress;
    Thread interpretThread = new Thread(() -> {
      try {
        InterpreterResult result1 = interpreter.interpret(
                "val df = sc.parallelize(1 to 10, 5).foreach(e=>Thread.sleep(1000))");
        assertEquals(InterpreterResult.Code.SUCCESS, result1.code());
      } catch (InterpreterException e) {
        e.printStackTrace();
        fail();
      }
    });
    interpretThread.start();
    boolean nonZeroProgress = false;
    int progress = 0;
    while (interpretThread.isAlive()) {
      progress = interpreter.getProgress();
      assertTrue(progress >= 0);
      if (progress != 0 && progress != 100) {
        nonZeroProgress = true;
      }
      Thread.sleep(100);
    }
    assertTrue(nonZeroProgress);

    interpretThread = new Thread(() -> {
      try {
        InterpreterResult result12 = interpreter.interpret(
                "val df = sc.parallelize(1 to 10, 2).foreach(e=>Thread.sleep(1000))");
        assertEquals(InterpreterResult.Code.ERROR, result12.code());
        assertTrue(result12.message().get(0).getData().contains("cancelled"));
      } catch (org.apache.submarine.interpreter.InterpreterException e) {
        e.printStackTrace();
        fail();
      }
    });

    interpretThread.start();
    // sleep 1 second to wait for the spark job start
    Thread.sleep(1000);
    interpreter.cancel();
    interpretThread.join();
  }

  @Test
  public void testDisableReplOutput() throws InterpreterException {
    Properties properties = new Properties();
    properties.setProperty("spark.master", "local");
    properties.setProperty("spark.app.name", "test");
    properties.setProperty("submarine.spark.maxResult", "100");
    properties.setProperty("submarine.spark.test", "true");
    properties.setProperty("submarine.spark.printREPLOutput", "false");
    // disable color output for easy testing
    properties.setProperty("submarine.spark.scala.color", "false");
    properties.setProperty("submarine.spark.deprecatedMsg.show", "false");

    interpreter = new SparkInterpreter(properties);
    interpreter.open();

    InterpreterResult result = interpreter.interpret("val a=\"hello world\"");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    // no output for define new variable
    assertEquals("", output);

    result = interpreter.interpret("print(a)");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    // output from print statement will still be displayed
    assertEquals("hello world", result.message().get(0).getData());
  }

  @Test
  public void testSchedulePool() throws InterpreterException {
    Properties properties = new Properties();
    properties.setProperty("spark.master", "local");
    properties.setProperty("spark.app.name", "test");
    properties.setProperty("submarine.spark.maxResult", "100");
    properties.setProperty("submarine.spark.test", "true");
    properties.setProperty("spark.scheduler.mode", "FAIR");
    // disable color output for easy testing
    properties.setProperty("submarine.spark.scala.color", "false");
    properties.setProperty("submarine.spark.deprecatedMsg.show", "false");
    properties.setProperty("spark.scheduler.pool", "pool1");

    interpreter = new SparkInterpreter(properties);
    interpreter.setSchedulerPool("pool1");
    interpreter.open();

    InterpreterResult result = interpreter.interpret("sc.range(1, 10).sum");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
    assertEquals("pool1", interpreter.getSparkContext().getLocalProperty("spark.scheduler.pool"));
  }

  // spark.ui.enabled: false
  @Test
  public void testDisableSparkUI_1() throws InterpreterException {
    Properties properties = new Properties();
    properties.setProperty("spark.master", "local");
    properties.setProperty("spark.app.name", "test");
    properties.setProperty("submarine.spark.maxResult", "100");
    properties.setProperty("submarine.spark.test", "true");
    properties.setProperty("spark.ui.enabled", "false");
    // disable color output for easy testing
    properties.setProperty("submarine.spark.scala.color", "false");
    properties.setProperty("submarine.spark.deprecatedMsg.show", "false");

    interpreter = new SparkInterpreter(properties);
    interpreter.open();

    InterpreterResult result = interpreter.interpret("sc.range(1, 10).sum");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());

  }

  // submarine.spark.ui.hidden: true
  @Test
  public void testDisableSparkUI_2() throws InterpreterException {
    Properties properties = new Properties();
    properties.setProperty("spark.master", "local");
    properties.setProperty("spark.app.name", "test");
    properties.setProperty("submarine.spark.maxResult", "100");
    properties.setProperty("submarine.spark.test", "true");
    properties.setProperty("submarine.spark.ui.hidden", "true");
    // disable color output for easy testing
    properties.setProperty("submarine.spark.scala.color", "false");
    properties.setProperty("submarine.spark.deprecatedMsg.show", "false");

    interpreter = new SparkInterpreter(properties);
    interpreter.open();

    InterpreterResult result = interpreter.interpret("sc.range(1, 10).sum");
    assertEquals(InterpreterResult.Code.SUCCESS, result.code());
  }


  @After
  public void tearDown() throws InterpreterException {
    if (this.interpreter != null) {
      this.interpreter.close();
    }
  }
}
