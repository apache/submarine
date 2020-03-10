/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.spark.security

import org.apache.spark.sql.hive.test.TestHive
import org.apache.spark.sql.SubmarineSparkUtils._
import org.apache.spark.sql.catalyst.plans.logical.{Project, SubmarineRowFilter}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class RowFilterSQLTest extends FunSuite with BeforeAndAfterAll {

  private val spark = TestHive.sparkSession.newSession()
  private lazy val sql = spark.sql _

  override def beforeAll(): Unit = {
    super.beforeAll()

    sql(
      """
        |CREATE TABLE IF NOT EXISTS default.rangertbl1 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE IF NOT EXISTS default.rangertbl2 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE IF NOT EXISTS default.rangertbl3 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE IF NOT EXISTS default.rangertbl4 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE IF NOT EXISTS default.rangertbl5 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE IF NOT EXISTS default.rangertbl6 AS SELECT * FROM default.src
      """.stripMargin)
    enableRowFilter(spark)
  }

  override def afterAll(): Unit = {
    super.afterAll()
    spark.reset()
  }


  test("simple query") {
    val statement = "select * from default.src"
    withUser("bob") {
      val df = sql(statement)
      assert(df.queryExecution.optimizedPlan.find(_.isInstanceOf[SubmarineRowFilter]).nonEmpty)
      val row = df.take(1)(0)
      assert(row.getInt(0) < 20, "keys above 20 should be filtered automatically")
      assert(df.count() === 20, "keys above 20 should be filtered automatically")
    }
    withUser("alice") {
      val df = sql(statement)
      assert(df.count() === 500)
    }
  }

  test("projection with ranger filter key") {
    val statement = "select key from default.src"
    withUser("bob") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getInt(0) < 20)
    }
    withUser("alice") {
      val df = sql(statement)
      assert(df.count() === 500)
    }
  }

  test("projection without ranger filter key") {
    val statement = "select value from default.src"
    withUser("bob") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getString(0).split("_")(1).toInt < 20)
    }
    withUser("alice") {
      val df = sql(statement)
      assert(df.count() === 500)
    }
  }

  test("filter with with ranger filter key") {
    val statement = "select key from default.src where key = 0"
    val statement2 = "select key from default.src where key >= 20"
    withUser("bob") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getInt(0) === 0)
      val df2 = sql(statement2)
      assert(df2.count() === 0, "all keys should be filtered")
    }
    withUser("alice") {
      val df = sql(statement)
      assert(df.count() === 3)
      val df2 = sql(statement2)
      assert(df2.count() === 480)
    }
  }

  test("WITH alias") {
    val statement = "select key as k1, value v1 from default.src"
    withUser("bob") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getInt(0) < 20, "keys above 20 should be filtered automatically")
      assert(df.count() === 20, "keys above 20 should be filtered automatically")
    }
    withUser("alice") {
      val df = sql(statement)
      assert(df.count() === 500)
    }
  }

  test("aggregate") {
    val statement = "select sum(key) as k1, value v1 from default.src group by v1"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1).split("_")(1).toInt < 20)
    }
    withUser("alice") {
      val df = sql(statement)
      assert(df.count() === 309)
    }
  }

  test("with equal expression") {
    val statement = "select * from default.rangertbl1"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getInt(0) === 0, "rangertbl1 has an internal expression key=0")
    }
  }

  test("with in set") {
    val statement = "select * from default.rangertbl2"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getInt(0) === 0, "rangertbl2 has an internal expression key in (0, 1, 2)")
    }
  }

  test("with in subquery") {
    val statement = "select * from default.rangertbl3"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val rows = df.collect()
      assert(rows.forall(_.getInt(0) < 100), "rangertbl3 has an internal expression key in (query)")
    }
  }

  test("with in subquery self joined") {
    val statement = "select * from default.rangertbl4"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val rows = df.collect()
      assert(rows.length === 500)
    }
  }

  test("with udf") {
    val statement = "select * from default.rangertbl5"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val rows = df.collect()
      assert(rows.length === 0)
    }
  }

  test("with multiple expressions") {
    val statement = "select * from default.rangertbl6"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val rows = df.collect()
      assert(rows.forall { r => val x = r.getInt(0); x > 1 && x < 10 || x == 500 })
    }
  }
}
