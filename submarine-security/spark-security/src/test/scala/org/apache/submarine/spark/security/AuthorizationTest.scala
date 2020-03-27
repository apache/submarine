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

import java.util.NoSuchElementException

import org.apache.spark.sql.execution.command.SubmarineResetCommand
import org.apache.spark.sql.hive.test.TestHive
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class AuthorizationTest extends FunSuite with BeforeAndAfterAll {

  import org.apache.spark.sql.SubmarineSparkUtils._
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
        |CREATE DATABASE testdb
        |""".stripMargin)
    // before authorization enabled
    withUser("alice") {
      assert(sql("show databases").count() === 2)
    }
    withUser("bob") {
      assert(sql("show databases").count() === 2)
    }
    withUser("kent") {
      assert(sql("show databases").count() === 2)
    }

    withUser("alice") {
      assert(sql("show tables").count() === 3)
    }
    withUser("bob") {
      assert(sql("show tables").count() === 3)
    }

    enableAuthorizer(spark)
  }

  override def afterAll(): Unit = {
    super.afterAll()
    spark.reset()
  }

  test("reset command") {
    val sparkConf = spark.sparkContext.getConf
    sql("set submarine.spark.some=any")
    assert(spark.sessionState.conf.getConfString("submarine.spark.some") === "any")
    val reset = sql("reset")

    assert(reset.queryExecution.optimizedPlan.getClass === SubmarineResetCommand.getClass)

    intercept[NoSuchElementException] {
      spark.sessionState.conf.getConfString("submarine.spark.some")
    }

    assert(spark.sessionState.conf.getConfString("spark.ui.enabled") ===
      sparkConf.get("spark.ui.enabled"))

    assert(spark.sessionState.conf.getConfString("spark.app.id") ===
      sparkConf.getAppId)
  }

  test("show databases") {
    withUser("alice") {
      assert(sql("show databases").count() === 0)
    }
    withUser("bob") {
      assert(sql("show databases").count() === 1)
      assert(sql("show databases").head().getString(0) === "default")
    }
    withUser("kent") {
      assert(sql("show databases").count() === 1)
    }
  }

  test("show tables") {
    withUser("alice") {
      assert(sql("show tables").count() === 0)
    }
    withUser("bob") {
      assert(sql("show tables").count() === 3)
    }
  }

  test("use database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("use default"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [USE] privilege" +
        " on [default]")
    }
    withUser("bob") {
      sql("use default")
    }
    withUser("kent") {
      sql("use default")
    }
  }

  test("create database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("create database db1"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [CREATE] privilege" +
        " on [db1]")
    }
  }

  test("describe database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("desc database default"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [USE] privilege on" +
        " [default]")
    }

    withUser("bob") {
      sql("desc database default")
    }
  }

  test("drop database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("drop database testdb"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [DROP] privilege" +
        " on [testdb]")
    }

    withUser("admin") {
      sql("drop database testdb")
    }
  }

  test("create table") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("create table default.alice(key int)"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [CREATE] privilege" +
        " on [default/alice]")
    }

    withUser("bob") {
      sql("create table default.bob(key int)")
    }
  }

  test("alter table") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException] {
        sql("alter table default.src set tblproperties('abc'='xyz')")
      }
      assert(e.getMessage === "Permission denied: user [alice] does not have [ALTER] privilege" +
        " on [default/src]")
    }

    withUser("bob") {
      sql("alter table default.src set tblproperties('abc'='xyz')")
    }
  }

  test("drop table") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("drop table default.rangertbl1"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [DROP] privilege" +
        " on [default/rangertbl1]")
    }

    withUser("bob") {
      sql("drop table default.rangertbl1")
    }
  }

  test("select") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("select * from default.rangertbl2").head())
      assert(e.getMessage === "Permission denied: user [alice] does not have [SELECT] privilege" +
        " on [default/rangertbl2/key,value]")
    }

    withUser("bob") {
      sql("select * from default.src").head()
    }

    withUser("kent") {
      sql("select key from default.src").head()
    }
    withUser("kent") {
      val e = intercept[SparkAccessControlException](sql("select value from default.src").head())
      assert(e.getMessage === "Permission denied: user [kent] does not have [SELECT] privilege" +
        " on [default/src/value]")
    }
    withUser("kent") {
      val e = intercept[SparkAccessControlException](sql("select * from default.src").head())
      assert(e.getMessage === "Permission denied: user [kent] does not have [SELECT] privilege" +
        " on [default/src/key,value]")
    }
  }
}
