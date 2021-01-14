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

package org.apache.submarine.spark.security

import org.apache.commons.codec.digest.DigestUtils
import org.apache.spark.sql.SubmarineSparkUtils.{enableDataMasking, withUser}
import org.apache.spark.sql.catalyst.plans.logical.{GlobalLimit, Project, SubmarineDataMasking}
import org.apache.spark.sql.hive.test.TestHive
import org.scalatest.{BeforeAndAfterAll, FunSuite}

case class DataMaskingSQLTest() extends FunSuite with BeforeAndAfterAll {
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
    enableDataMasking(spark)
  }

  override def afterAll(): Unit = {
    super.afterAll()
    spark.reset()
  }

  test("simple query") {
    val statement = "select * from default.src"
    withUser("bob") {
      val df = sql(statement)
      assert(df.queryExecution.optimizedPlan.find(_.isInstanceOf[SubmarineDataMasking]).nonEmpty)
      assert(df.queryExecution.optimizedPlan.isInstanceOf[Project])
      val project = df.queryExecution.optimizedPlan.asInstanceOf[Project]
      val masker = project.projectList(1)
      assert(masker.name === "value")
      assert(masker.children.exists(_.sql.contains("mask_show_last_n")))
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("x"), "values should be masked")
    }
    withUser("alice") {
      assert(!sql(statement).take(1)(0).getString(1).startsWith("x"))
    }
  }

  test("projection with ranger filter key") {
    withUser("bob") {
      val statement = "select key from default.src where key = 0"
      val df = sql(statement)
      assert(df.queryExecution.optimizedPlan.find(_.isInstanceOf[SubmarineDataMasking]).nonEmpty)
      val row = df.take(1)(0)
      assert(row.getInt(0) === 0, "key is not masked")
    }
    withUser("bob") {
      val statement = "select value from default.src where key = 0"
      val df = sql(statement)
      assert(df.queryExecution.optimizedPlan.find(_.isInstanceOf[SubmarineDataMasking]).nonEmpty)
      val row = df.take(1)(0)
      assert(row.getString(0).startsWith("x"), "value is masked")
    }
  }

  test("alias") {
    val statement = "select key as k1, value v1 from default.src"
    withUser("bob") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("x"), "values should be masked")
    }
  }

  test("alias, non-alias coexists") {
    val statement = "select key as k1, value, value v1 from default.src"
    withUser("bob") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("x"), "values should be masked")
      assert(row.getString(2).startsWith("x"), "values should be masked")
    }
  }

  test("agg") {
    val statement = "select sum(key) as k1, value v1 from default.src group by v1"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("x"), "values should be masked")
    }
    withUser("alice") {
      val df = sql(statement)
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("val"), "values should not be masked")
    }
  }

  test("MASK") {
    val statement = "select * from default.rangertbl1"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("x"), "values should be masked")
    }
  }

  test("MASK_SHOW_FIRST_4") {
    val statement = "select * from default.rangertbl2"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("val_x"), "values should show first 4 characters")
    }
  }

  test("MASK_HASH") {
    val statement = "select * from default.rangertbl3 where value = 'val_277'"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === DigestUtils.md5Hex("val_277"), "value is hashed")
    }
  }

  test("MASK_NULL") {
    val statement = "select * from default.rangertbl4 where value = 'val_277'"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === null, "value is hashed")
    }
  }

  test("MASK_SHOW_LAST_4") {
    val statement = "select * from default.rangertbl5 where value = 'val_277'"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === "xxx_277", "value shows last 4 characters")
    }
  }

  test("MASK_SHOW_LAST_4 and functions") {
    val statement =
      s"""
         |select
         | key,
         | value,
         | substr(value, 0, 18),
         | substr(value, 0, 18) as v1,
         | substr(cast(value as string), 0, 18) as v2
         | from default.rangertbl5 where value = 'val_277'""".stripMargin
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === "xxx_277", "value shows last 4 characters")
      assert(row.getString(2) === "xxx_277", "value shows last 4 characters")
      assert(row.getString(3) === "xxx_277", "value shows last 4 characters")
      assert(row.getString(4) === "xxx_277", "value shows last 4 characters")

    }
  }

  test("NO MASKING") {
    val statement = "select * from default.rangertbl6 where value = 'val_277'"
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === "val_277", "value has no mask")
    }
  }

  test("commands") {
    withUser("bob") {
      val statement = "create view v1 as select * from default.rangertbl5 where value = 'val_277'"
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)

      val row = sql("select * from v1").take(1)(0)
      assert(row.getString(1) === "xxx_277", "value shows last 4 characters")
    }
  }

  test("MASK_SHOW_LAST_4 with uncorrelated subquery") {
    val statement =
      s"""
         |select
         | *
         |from default.rangertbl5 outer
         |where value in (select value from default.rangertbl4 where value = 'val_277')
         |""".stripMargin
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === "xxx_277", "value shows last 4 characters")
    }
  }

  test("MASK_SHOW_LAST_4 with correlated subquery") {
    val statement =
      s"""
         |select
         | *
         |from default.rangertbl5 outer
         |where key =
         | (select max(key) from default.rangertbl4 where value = 'val_277' and value = outer.value)
         |""".stripMargin
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(1) === "xxx_277", "value shows last 4 characters")
    }
  }

  test("CTE") {
    val statement =
      s"""
         |with myCTE as
         |(select
         | *
         |from default.rangertbl5 where value = 'val_277')
         |select t1.value, t2.value from myCTE t1 join myCTE t2 on t1.key = t2.key
         |
         |""".stripMargin
    withUser("bob") {
      val df = sql(statement)
      println(df.queryExecution.optimizedPlan)
      val row = df.take(1)(0)
      assert(row.getString(0) === "xxx_277", "value shows last 4 characters")
      assert(row.getString(1) === "xxx_277", "value shows last 4 characters")
    }
  }

  test("query limit expression") {
    withUser("bob") {
      val df = sql("select * from default.src limit 10")
      assert(df.queryExecution.optimizedPlan.find(_.isInstanceOf[SubmarineDataMasking]).nonEmpty)
      assert(df.queryExecution.optimizedPlan.isInstanceOf[GlobalLimit])
      val row = df.take(1)(0)
      assert(row.getString(1).startsWith("x"), "values should be masked")
    }
  }
}
