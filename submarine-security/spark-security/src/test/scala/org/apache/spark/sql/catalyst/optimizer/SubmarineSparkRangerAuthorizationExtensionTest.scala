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

package org.apache.spark.sql.catalyst.optimizer

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.apache.spark.sql.execution.command.{CreateDatabaseCommand, ShowDatabasesCommand, ShowTablesCommand}
import org.apache.spark.sql.execution.{SubmarineShowDatabasesCommand, SubmarineShowTablesCommand}
import org.apache.spark.sql.hive.test.TestHive
import org.apache.submarine.spark.security.SparkAccessControlException

class SubmarineSparkRangerAuthorizationExtensionTest extends FunSuite with BeforeAndAfterAll {

  private val spark = TestHive.sparkSession.newSession()

  private val authz = SubmarineSparkRangerAuthorizationExtension(spark)

  test("replace submarine show databases") {
    val df = spark.sql("show databases")
    val originalPlan = df.queryExecution.optimizedPlan
    assert(originalPlan.isInstanceOf[ShowDatabasesCommand])
    val newPlan = authz(originalPlan)
    assert(newPlan.isInstanceOf[SubmarineShowDatabasesCommand])
  }

  test("replace submarine show tables") {
    val df = spark.sql("show tables")
    val originalPlan = df.queryExecution.optimizedPlan
    assert(originalPlan.isInstanceOf[ShowTablesCommand])
    val newPlan = authz(originalPlan)
    assert(newPlan.isInstanceOf[SubmarineShowTablesCommand])
  }

  test("fail to create database by default") {
    try {
      val df = spark.sql("create database testdb1")
      val originalPlan = df.queryExecution.optimizedPlan
      assert(originalPlan.isInstanceOf[CreateDatabaseCommand])
      intercept[SparkAccessControlException](authz(originalPlan))
    } finally {
      spark.sql("drop database testdb1")
    }
  }

}
