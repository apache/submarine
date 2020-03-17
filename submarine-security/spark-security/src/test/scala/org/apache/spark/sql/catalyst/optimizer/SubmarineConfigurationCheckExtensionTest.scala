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

package org.apache.spark.sql.catalyst.optimizer

import org.apache.spark.sql.hive.test.TestHive
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.submarine.spark.security.SparkAccessControlException

class SubmarineConfigurationCheckExtensionTest extends FunSuite with BeforeAndAfterAll{

  private val spark = TestHive.sparkSession.newSession()

  override def afterAll(): Unit = {
    super.afterAll()
    spark.reset()
  }

  test("apply spark configuration restriction rules") {
    spark.sql("set spark.sql.submarine.conf.restricted.list=spark.sql.abc,spark.sql.xyz")
    val extension = SubmarineConfigurationCheckExtension(spark)
    val p1 = spark.sql("set spark.sql.runSQLOnFiles=true").queryExecution.optimizedPlan
    intercept[SparkAccessControlException](extension.apply(p1))
    val p2 = spark.sql("set spark.sql.runSQLOnFiles=false").queryExecution.optimizedPlan
    intercept[SparkAccessControlException](extension.apply(p2))
    val p3 = spark.sql("set spark.sql.runSQLOnFiles;").queryExecution.optimizedPlan
    extension.apply(p3)
    val p4 = spark.sql("set spark.sql.abc=xyz").queryExecution.optimizedPlan
    intercept[SparkAccessControlException](extension.apply(p4))
    val p5 = spark.sql("set spark.sql.xyz=abc").queryExecution.optimizedPlan
    intercept[SparkAccessControlException](extension.apply(p5))
    val p6 = spark.sql("set spark.sql.submarine.conf.restricted.list=123")
      .queryExecution.optimizedPlan
    intercept[SparkAccessControlException](extension.apply(p6))
    val p7 = spark.sql("set spark.sql.efg=hijk;").queryExecution.optimizedPlan
    extension.apply(p7)
  }
}
