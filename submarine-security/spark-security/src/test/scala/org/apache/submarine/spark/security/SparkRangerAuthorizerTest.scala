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
import org.apache.spark.sql.internal.SQLConf
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SparkRangerAuthorizerTest extends FunSuite with BeforeAndAfterAll {

  import org.apache.spark.sql.RangerSparkTestUtils._
  private val spark = TestHive.sparkSession
  private lazy val sql = spark.sql _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark.conf.set(SQLConf.CROSS_JOINS_ENABLED.key, "true")

    sql(
      """
        |CREATE TABLE default.rangertbl1 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE default.rangertbl2 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE default.rangertbl3 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE default.rangertbl4 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE default.rangertbl5 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE TABLE default.rangertbl6 AS SELECT * FROM default.src
      """.stripMargin)

    sql(
      """
        |CREATE DATABASE testdb
        |""".stripMargin)
    enableAuthorizer(spark)
  }

  test("use database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("use default"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [USE] privilege on [default]")
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
      assert(e.getMessage === "Permission denied: user [alice] does not have [CREATE] privilege on [db1]")
    }
  }

  test("describe database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("desc database default"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [USE] privilege on [default]")
    }

    withUser("bob") {
      sql("desc database default")
    }
  }

  test("drop database") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("drop database testdb"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [DROP] privilege on [testdb]")
    }

    withUser("admin") {
      sql("drop database testdb")
    }
  }

  test("create table") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("create table default.alice(key int)"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [CREATE] privilege on [default/alice]")
    }

    withUser("bob") {
      sql("create table default.bob(key int)")
    }
  }

  test("alter table") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("alter table default.src set tblproperties('abc'='xyz')"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [ALTER] privilege on [default/src]")
    }

    withUser("bob") {
      sql("alter table default.src set tblproperties('abc'='xyz')")
    }
  }

  test("drop table") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("drop table default.rangertbl1"))
      assert(e.getMessage === "Permission denied: user [alice] does not have [DROP] privilege on [default/rangertbl1]")
    }

    withUser("bob") {
      sql("drop table default.rangertbl1")
    }
  }

  test("select") {
    withUser("alice") {
      val e = intercept[SparkAccessControlException](sql("select * from default.rangertbl2").head())
      assert(e.getMessage === "Permission denied: user [alice] does not have [SELECT] privilege on [default/rangertbl2/key,value]")
    }

    withUser("bob") {
      sql("select * from default.src").head()
    }

    withUser("kent") {
      sql("select key from default.src").head()
    }
    withUser("kent") {
      val e = intercept[SparkAccessControlException](sql("select value from default.src").head())
      assert(e.getMessage === "Permission denied: user [kent] does not have [SELECT] privilege on [default/src/value]")
    }
    withUser("kent") {
      val e = intercept[SparkAccessControlException](sql("select * from default.src").head())
      assert(e.getMessage === "Permission denied: user [kent] does not have [SELECT] privilege on [default/src/key,value]")
    }
  }
}
