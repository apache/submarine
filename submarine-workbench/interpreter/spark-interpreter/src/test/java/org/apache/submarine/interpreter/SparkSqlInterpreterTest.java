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

import org.apache.zeppelin.display.AngularObjectRegistry;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterException;
import org.apache.zeppelin.interpreter.InterpreterGroup;
import org.apache.zeppelin.interpreter.InterpreterOutput;
import org.apache.zeppelin.interpreter.remote.RemoteInterpreterEventClient;
import org.apache.zeppelin.resource.LocalResourcePool;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.LinkedList;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;

public class SparkSqlInterpreterTest {

  private static SparkSqlInterpreter sqlInterpreter;
  private static SparkInterpreter sparkInterpreter;
  private static InterpreterContext context;
  private static InterpreterGroup intpGroup;


  @BeforeClass
  public static void setUp() throws InterpreterException {
    Properties p = new Properties();
    p.setProperty("spark.master", "local[4]");
    p.setProperty("spark.app.name", "test");
    p.setProperty("zeppelin.spark.maxResult", "10");
    p.setProperty("zeppelin.spark.concurrentSQL", "true");
    p.setProperty("zeppelin.spark.sql.stacktrace", "true");
    p.setProperty("zeppelin.spark.useHiveContext", "true");
    p.setProperty("zeppelin.spark.deprecatedMsg.show", "false");

    sqlInterpreter = new SparkSqlInterpreter(p);

    intpGroup = new InterpreterGroup();
    sparkInterpreter = new SparkInterpreter(p);
    sparkInterpreter.setInterpreterGroup(intpGroup);

    sqlInterpreter = new SparkSqlInterpreter(p);
    sqlInterpreter.setInterpreterGroup(intpGroup);

    String session = "session_1";
    intpGroup.put(session, new LinkedList<>());
    sparkInterpreter.addToSession(session);
    sqlInterpreter.addToSession(session);

    context = InterpreterContext.builder()
            .setNoteId("noteId")
            .setParagraphId("paragraphId")
            .setParagraphTitle("title")
            .setAngularObjectRegistry(new AngularObjectRegistry(intpGroup.getId(), null))
            .setResourcePool(new LocalResourcePool("id"))
            .setInterpreterOut(new InterpreterOutput(null))
            .setIntpEventClient(mock(RemoteInterpreterEventClient.class))
            .build();

    sparkInterpreter.setIntpContext(context);
    sqlInterpreter.setIntpContext(context);

    //SparkInterpreter will change the current thread's classLoader
    Thread.currentThread().setContextClassLoader(SparkInterpreterTest.class.getClassLoader());

    sparkInterpreter.open();
    sqlInterpreter.open();
  }

  @AfterClass
  public static void tearDown() throws InterpreterException {
    sqlInterpreter.close();
  }

  @Test
  public void test() {
    sparkInterpreter.interpret("case class Test(name:String, age:Int)");
    sparkInterpreter.interpret(
            "val test = sc.parallelize(Seq(" +
                    " Test(\"moon\", 33)," +
                    " Test(\"jobs\", 51)," +
                    " Test(\"gates\", 51)," +
                    " Test(\"park\", 34)" +
                    "))");
    sparkInterpreter.interpret("test.toDF.registerTempTable(\"test\")");

    InterpreterResult ret = sqlInterpreter.interpret("select name, age from test where age < 40");
    assertEquals(InterpreterResult.Code.SUCCESS, ret.code());
    assertEquals(org.apache.submarine.interpreter.InterpreterResult.Type.TABLE,
                 ret.message().get(0).getType()
    );
    assertEquals("name\tage\nmoon\t33\npark\t34\n", ret.message().get(0).getData());

    ret = sqlInterpreter.interpret("select wrong syntax");
    assertEquals(InterpreterResult.Code.ERROR, ret.code());
    assertTrue(ret.message().get(0).getData().length() > 0);

    assertEquals(InterpreterResult.Code.SUCCESS,
                 sqlInterpreter.interpret("select case when name='aa' then name else name end from test")
                         .code()
    );
  }

  @Test
  public void testStruct() {
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

    InterpreterResult ret = sqlInterpreter.interpret("select * from gr");
    assertEquals(InterpreterResult.Code.SUCCESS, ret.code());

  }


  @Test
  public void testMaxResults() {
    sparkInterpreter.interpret("case class P(age:Int)");
    sparkInterpreter.interpret(
            "val gr = sc.parallelize(Seq(P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(8),P(9),P(10),P(11)))");
    sparkInterpreter.interpret("gr.toDF.registerTempTable(\"gr\")");

    InterpreterResult ret = sqlInterpreter.interpret("select * from gr");
    assertEquals(InterpreterResult.Code.SUCCESS, ret.code());
    // the number of rows is 10+1, 1 is the head of table
    assertEquals(11, ret.message().get(0).getData().split("\n").length);
    assertTrue(ret.message().get(1).getData().contains("alert-warning"));

    // test limit local property
    context.getLocalProperties().put("limit", "5");
    ret = sqlInterpreter.interpret("select * from gr");
    assertEquals(InterpreterResult.Code.SUCCESS, ret.code());
    // the number of rows is 5+1, 1 is the head of table
    assertEquals(6, ret.message().get(0).getData().split("\n").length);
  }

  @Test
  public void testConcurrentSQL() throws InterruptedException {

    sparkInterpreter.interpret("spark.udf.register(\"sleep\", (e:Int) => {Thread.sleep(e*1000); e})");


    Thread thread1 = new Thread() {
      @Override
      public void run() {
        InterpreterResult result = sqlInterpreter.interpret("select sleep(10)");
        assertEquals(InterpreterResult.Code.SUCCESS, result.code());
      }
    };

    Thread thread2 = new Thread() {
      @Override
      public void run() {
        InterpreterResult result = sqlInterpreter.interpret("select sleep(10)");
        assertEquals(InterpreterResult.Code.SUCCESS, result.code());
      }
    };

    // start running 2 spark sql, each would sleep 10 seconds, the totally running time should
    // be less than 20 seconds, which means they run concurrently.
    long start = System.currentTimeMillis();
    thread1.start();
    thread2.start();
    thread1.join();
    thread2.join();
    long end = System.currentTimeMillis();
    assertTrue("running time must be less than 20 seconds", ((end - start) / 1000) < 20);

  }

  @Test
  public void testDDL() {
    InterpreterResult ret = sqlInterpreter.interpret("create table t1(id int, name string)");
    assertEquals(InterpreterResult.Code.SUCCESS, ret.code());
    // spark 1.x will still return DataFrame with non-empty columns.
    // org.apache.spark.sql.DataFrame = [result: string]

    assertTrue(ret.message().isEmpty());

    // create the same table again
    ret = sqlInterpreter.interpret("create table t1(id int, name string)");
    assertEquals(InterpreterResult.Code.ERROR, ret.code());
    assertEquals(1, ret.message().size());
    assertEquals(org.apache.submarine.interpreter.InterpreterResult.Type.TEXT,
                 ret.message().get(0).getType()
    );
    assertTrue(ret.message().get(0).getData().contains("already exists"));

    // invalid DDL
    ret = sqlInterpreter.interpret("create temporary function udf1 as 'org.apache.zeppelin.UDF'");
    assertEquals(InterpreterResult.Code.ERROR, ret.code());
    assertEquals(1, ret.message().size());
    assertEquals(org.apache.submarine.interpreter.InterpreterResult.Type.TEXT,
                 ret.message().get(0).getType()
    );

  }
}
