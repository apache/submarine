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

import java.util.Locale

import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

import org.apache.submarine.spark.security.command.CreateRoleCommand
import org.apache.submarine.spark.security.parser.SubmarineSqlBaseParser.{CreateRoleContext, SingleStatementContext}

class SubmarineSqlAstBuilder extends SubmarineSqlBaseBaseVisitor[AnyRef] {

  override def visitSingleStatement(ctx: SingleStatementContext): LogicalPlan = {
    visit(ctx.statement()).asInstanceOf[LogicalPlan]
  }

  override def visitCreateRole(ctx: CreateRoleContext): AnyRef = {
    CreateRoleCommand(ctx.identifier().getText)
  }
}
