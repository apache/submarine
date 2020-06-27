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

import org.apache.spark.sql.SubmarineSparkUtils
import org.apache.spark.sql.catalyst.plans.logical.{Filter, SubmarineRowFilter}
import org.apache.spark.sql.hive.test.TestHive
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SubmarineRowFilterExtensionTest extends FunSuite with BeforeAndAfterAll {

  private val spark = TestHive.sparkSession.newSession()

  override def afterAll(): Unit = {
    super.afterAll()
    spark.reset()
  }

  test("applying condition to original query if row filter exists in ranger") {
    val extension = SubmarineRowFilterExtension(spark)
    val frame = spark.sql("select * from src")
    SubmarineSparkUtils.withUser("bob") {
      val plan = extension.apply(frame.queryExecution.optimizedPlan)
      assert(plan.collect { case f: Filter => f }.nonEmpty)
      assert(plan.isInstanceOf[SubmarineRowFilter])
    }

    SubmarineSparkUtils.withUser("alice") {
      val plan = extension.apply(frame.queryExecution.optimizedPlan)
      assert(plan.collect { case f: Filter => f }.isEmpty)
      assert(plan.isInstanceOf[SubmarineRowFilter])
    }
  }
}
