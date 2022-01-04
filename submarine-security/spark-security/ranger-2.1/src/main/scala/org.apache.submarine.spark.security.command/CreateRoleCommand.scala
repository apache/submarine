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

package org.apache.submarine.spark.security.command

import org.apache.hadoop.security.UserGroupInformation
import org.apache.ranger.plugin.model.RangerRole
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.command.LeafRunnableCommand
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.submarine.spark.security.{RangerSparkAuditHandler, RangerSparkPlugin, SparkAccessControlException}

import java.util.Arrays
import scala.util.control.NonFatal


case class CreateRoleCommand(roleName: String) extends LeafRunnableCommand {
  import CommandUtils._
  override def run(sparkSession: SparkSession): Seq[Row] = {

    validateRoleName(roleName)
    val auditHandler = RangerSparkAuditHandler()
    val currentUser = UserGroupInformation.getCurrentUser.getShortUserName

    val role = new RangerRole()
    role.setName(roleName)
    role.setCreatedByUser(currentUser)
    role.setCreatedBy(currentUser)
    role.setUpdatedBy(currentUser)
    val member = new RangerRole.RoleMember(currentUser, true)
    role.setUsers(Arrays.asList(member))
    try {
      val res = RangerSparkPlugin.createRole(role, auditHandler)
      logDebug(s"Create role: ${res.getName} success")
      Seq.empty[Row]
    } catch {
      case NonFatal(e) => throw new SparkAccessControlException(e.getMessage, e)
    } finally {
      // TODO: support auditHandler.flushAudit()
    }
  }
}
