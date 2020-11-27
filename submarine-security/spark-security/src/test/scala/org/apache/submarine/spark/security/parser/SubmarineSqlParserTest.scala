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

package org.apache.submarine.spark.security.parser

import org.apache.spark.sql.catalyst.parser.ParseException
import org.apache.spark.sql.hive.test.TestHive
import org.scalatest.FunSuite

import org.apache.submarine.spark.security.command.{CreateRoleCommand, DropRoleCommand, ShowCurrentRolesCommand, ShowRolesCommand}

class SubmarineSqlParserTest extends FunSuite {

  private val spark = TestHive.sparkSession.newSession()

  val parser = new SubmarineSqlParserCompatible(spark.sessionState.sqlParser)

  test("create role") {
    val p1 = parser.parsePlan("create role abc")
    assert(p1.isInstanceOf[CreateRoleCommand])
    assert(p1.asInstanceOf[CreateRoleCommand].roleName === "abc")
    val p2 = parser.parsePlan("create role admin")
    assert(p2.isInstanceOf[CreateRoleCommand])
    assert(p2.asInstanceOf[CreateRoleCommand].roleName === "admin")
    val p3 = parser.parsePlan("create role `bob`")
    assert(p3.isInstanceOf[CreateRoleCommand])
    assert(p3.asInstanceOf[CreateRoleCommand].roleName === "`bob`")
    intercept[ParseException](parser.parsePlan("create role 'bob'"))
  }

  test("drop role") {
    val p1 = parser.parsePlan("drop role abc")
    assert(p1.isInstanceOf[DropRoleCommand])
    assert(p1.asInstanceOf[DropRoleCommand].roleName === "abc")
    val p2 = parser.parsePlan("drop role admin")
    assert(p2.isInstanceOf[DropRoleCommand])
    assert(p2.asInstanceOf[DropRoleCommand].roleName === "admin")
    val p3 = parser.parsePlan("drop role `bob`")
    assert(p3.isInstanceOf[DropRoleCommand])
    assert(p3.asInstanceOf[DropRoleCommand].roleName === "`bob`")
    intercept[ParseException](parser.parsePlan("drop role 'bob'"))
  }

  test("show roles") {
    val p1 = parser.parsePlan("show roles")
    assert(p1.isInstanceOf[ShowRolesCommand])
  }

  test("show current roles") {
    val p1 = parser.parsePlan("show current roles")
    assert(p1.isInstanceOf[ShowCurrentRolesCommand])
  }
}
