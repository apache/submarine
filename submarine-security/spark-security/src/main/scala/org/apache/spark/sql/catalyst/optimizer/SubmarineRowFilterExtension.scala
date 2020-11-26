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

import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.security.UserGroupInformation
import org.apache.ranger.plugin.policyengine.RangerAccessResult
import org.apache.spark.sql.AuthzUtils.getFieldVal
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.catalog.CatalogTable
import org.apache.spark.sql.catalyst.expressions.SubqueryExpression
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.command.{CreateDataSourceTableAsSelectCommand, CreateViewCommand, InsertIntoDataSourceDirCommand}
import org.apache.spark.sql.execution.datasources.{InsertIntoDataSourceCommand, InsertIntoHadoopFsRelationCommand, LogicalRelation, SaveIntoDataSourceCommand}
import org.apache.spark.sql.hive.execution.{CreateHiveTableAsSelectCommand, InsertIntoHiveDirCommand, InsertIntoHiveTable}
import org.apache.submarine.spark.compatible.SubqueryCompatible
import org.apache.submarine.spark.security._

/**
 * An Apache Spark's [[Optimizer]] extension for row level filtering.
 */
case class SubmarineRowFilterExtension(spark: SparkSession) extends Rule[LogicalPlan] {
  private lazy val rangerSparkOptimizer = new SubmarineSparkOptimizer(spark)

  /**
   * Transform a Relation to a parsed [[LogicalPlan]] with specified row filter expressions
   * @param plan the original [[LogicalPlan]]
   * @param table a Spark [[CatalogTable]] representation
   * @return A new Spark [[LogicalPlan]] with specified row filter expressions
   */
  private def applyingRowFilterExpr(plan: LogicalPlan, table: CatalogTable): LogicalPlan = {
    val auditHandler = RangerSparkAuditHandler()
    try {
      val identifier = table.identifier
      val resource =
        RangerSparkResource(SparkObjectType.TABLE, identifier.database, identifier.table)
      val ugi = UserGroupInformation.getCurrentUser
      val request = new RangerSparkAccessRequest(resource, ugi.getShortUserName,
        ugi.getGroupNames.toSet, SparkObjectType.TABLE.toString, SparkAccessType.SELECT,
        RangerSparkPlugin.getClusterName)
      val result = RangerSparkPlugin.evalRowFilterPolicies(request, auditHandler)
      if (isRowFilterEnabled(result)) {
        val condition = spark.sessionState.sqlParser.parseExpression(result.getFilterExpr)
        val analyzed = spark.sessionState.analyzer.execute(Filter(condition, plan))
        val optimized = analyzed transformAllExpressions {
          case s: SubqueryExpression =>
            val SubqueryCompatible(newPlan) =
              rangerSparkOptimizer.execute(SubqueryCompatible(SubmarineRowFilter(s.plan)))
            s.withNewPlan(newPlan)
        }
        SubmarineRowFilter(optimized)
      } else {
        SubmarineRowFilter(plan)
      }
    } catch {
      case e: Exception => throw e
    }
  }

  private def isRowFilterEnabled(result: RangerAccessResult): Boolean = {
    result != null && result.isRowFilterEnabled && StringUtils.isNotEmpty(result.getFilterExpr)
  }

  private def getPlanWithTables(plan: LogicalPlan): Map[LogicalPlan, CatalogTable] = {
    plan.collectLeaves().map {
      case h if h.nodeName == "HiveTableRelation" =>
        h -> getFieldVal(h, "tableMeta").asInstanceOf[CatalogTable]
      case m if m.nodeName == "MetastoreRelation" =>
        m -> getFieldVal(m, "catalogTable").asInstanceOf[CatalogTable]
      case l: LogicalRelation if l.catalogTable.isDefined =>
        l -> l.catalogTable.get
      case _ => null
    }.filter(_ != null).toMap
  }

  private def isFixed(plan: LogicalPlan): Boolean = {
    val markNum = plan.collect { case _: SubmarineRowFilter => true }.size
    markNum >= getPlanWithTables(plan).size
  }
  private def doFiltering(plan: LogicalPlan): LogicalPlan = plan match {
    case rf: SubmarineRowFilter => rf
    case plan if isFixed(plan) => plan
    case _ =>
      val plansWithTables = getPlanWithTables(plan)
        .map { case (plan, table) =>
          (plan, applyingRowFilterExpr(plan, table))
        }

      plan transformUp {
        case p => plansWithTables.getOrElse(p, p)
      }
  }

  /**
   * Transform a spark logical plan to another plan with the row filer expressions
   * @param plan the original [[LogicalPlan]]
   * @return the logical plan with row filer expressions applied
   */
  override def apply(plan: LogicalPlan): LogicalPlan = plan match {
    case c: Command => c match {
      case c: CreateDataSourceTableAsSelectCommand => c.copy(query = doFiltering(c.query))
      case c: CreateHiveTableAsSelectCommand => c.copy(query = doFiltering(c.query))
      case c: CreateViewCommand => c.copy(child = doFiltering(c.child))
      case i: InsertIntoDataSourceCommand => i.copy(query = doFiltering(i.query))
      case i: InsertIntoDataSourceDirCommand => i.copy(query = doFiltering(i.query))
      case i: InsertIntoHadoopFsRelationCommand => i.copy(query = doFiltering(i.query))
      case i: InsertIntoHiveDirCommand => i.copy(query = doFiltering(i.query))
      case i: InsertIntoHiveTable => i.copy(query = doFiltering(i.query))
      case s: SaveIntoDataSourceCommand => s.copy(query = doFiltering(s.query))
      case cmd => cmd
    }
    case other => doFiltering(other)
  }
}
