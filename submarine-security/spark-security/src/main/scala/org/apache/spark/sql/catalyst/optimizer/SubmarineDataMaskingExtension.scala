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

import scala.collection.mutable

import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.security.UserGroupInformation
import org.apache.ranger.plugin.policyengine.RangerAccessResult
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{FunctionIdentifier, TableIdentifier}
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.catalog.{CatalogFunction, CatalogTable, HiveTableRelation}
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference, Expression, ExprId, NamedExpression, SubqueryExpression}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.command.{CreateDataSourceTableAsSelectCommand, CreateViewCommand, InsertIntoDataSourceDirCommand}
import org.apache.spark.sql.execution.datasources.{InsertIntoDataSourceCommand, InsertIntoHadoopFsRelationCommand, LogicalRelation, SaveIntoDataSourceCommand}
import org.apache.spark.sql.hive.execution.{CreateHiveTableAsSelectCommand, InsertIntoHiveDirCommand, InsertIntoHiveTable}

import org.apache.submarine.spark.compatible.SubqueryCompatible
import org.apache.submarine.spark.security._
import org.apache.submarine.spark.security.SparkObjectType.COLUMN


/**
 * An Apache Spark's [[Optimizer]] extension for column data masking.
 * TODO(kent yao) implement this as analyzer rule
 */
case class SubmarineDataMaskingExtension(spark: SparkSession) extends Rule[LogicalPlan] {
  import org.apache.ranger.plugin.model.RangerPolicy._

  // register all built-in masking udfs
  Map("mask" -> "org.apache.hadoop.hive.ql.udf.generic.GenericUDFMask",
    "mask_first_n" -> "org.apache.hadoop.hive.ql.udf.generic.GenericUDFMaskFirstN",
    "mask_hash" -> "org.apache.hadoop.hive.ql.udf.generic.GenericUDFMaskHash",
    "mask_last_n" -> "org.apache.hadoop.hive.ql.udf.generic.GenericUDFMaskLastN",
    "mask_show_first_n" -> "org.apache.hadoop.hive.ql.udf.generic.GenericUDFMaskShowFirstN",
    "mask_show_last_n" -> "org.apache.hadoop.hive.ql.udf.generic.GenericUDFMaskShowLastN")
    .map(x => CatalogFunction(FunctionIdentifier(x._1), x._2, Seq.empty))
    .foreach(spark.sessionState.catalog.registerFunction(_, overrideIfExists = true))

  private lazy val sqlParser = spark.sessionState.sqlParser
  private lazy val analyzer = spark.sessionState.analyzer
  private lazy val auditHandler = RangerSparkAuditHandler()
  private def currentUser: UserGroupInformation = UserGroupInformation.getCurrentUser

  /**
   * Get RangerAccessResult from ranger admin or local policies, which contains data masking rules
   */
  private def getAccessResult(identifier: TableIdentifier, attr: Attribute): RangerAccessResult = {
    val resource = RangerSparkResource(COLUMN, identifier.database, identifier.table, attr.name)
    val req = new RangerSparkAccessRequest(
      resource,
      currentUser.getShortUserName,
      currentUser.getGroupNames.toSet,
      COLUMN.toString,
      SparkAccessType.SELECT,
      RangerSparkPlugin.getClusterName)
    RangerSparkPlugin.evalDataMaskPolicies(req, auditHandler)
  }

  /**
   * Generate an [[Alias]] expression with the access result and original expression, which can be
   * used to replace the original output of the query.
   *
   * This alias contains a child, which might be null literal or [[UnresolvedFunction]]. When the
   * child is function, it replace the argument which is [[UnresolvedAttribute]] with the input
   * attribute to resolve directly.
   */
  private def getMasker(attr: Attribute, result: RangerAccessResult): Alias = {
    val expr = if (StringUtils.equalsIgnoreCase(result.getMaskType, MASK_TYPE_NULL)) {
      "NULL"
    } else if (StringUtils.equalsIgnoreCase(result.getMaskType, MASK_TYPE_CUSTOM)) {
      val maskVal = result.getMaskedValue
      if (maskVal == null) {
        "NULL"
      } else {
        s"${maskVal.replace("{col}", attr.name)}"
      }
    } else if (result.getMaskTypeDef != null) {
      val transformer = result.getMaskTypeDef.getTransformer
      if (StringUtils.isNotEmpty(transformer)) {
        s"${transformer.replace("{col}", attr.name)}"
      } else {
        return null
      }
    } else {
      return null
    }

    // sql expression text -> UnresolvedFunction
    val parsed = sqlParser.parseExpression(expr)

    // Here we replace the attribute with a new resolved one, e.g.
    // 'mask_show_last_n('value, 4, x, x, x, -1, 1)
    // ->
    // 'mask_show_last_n(value#38, 4, x, x, x, -1, 1) AS value#37
    // value#38 will be pushed down to the relation and value#37 will be used for the associated parent node
    val resolved = parsed mapChildren {
      case u: UnresolvedAttribute => AttributeReference(attr.name, attr.dataType, attr.nullable, attr.metadata)(
        qualifier = attr.qualifier)
      case o => o
    }
    Alias(resolved, attr.name)(attr.exprId, attr.qualifier, Option(attr.metadata))
  }

  /**
   * Collecting transformers from Ranger data masking policies, and mapping the to the
   * [[LogicalPlan]] output attributes.
   *
   * @param plan the original logical plan with a underlying catalog table
   * @param table the catalog table
   * @return a list of key-value pairs of original expression with its masking representation
   */
  private def collectTransformers(
      plan: LogicalPlan,
      table: CatalogTable,
      aliases: mutable.Map[Alias, ExprId],
      outputs: Seq[NamedExpression]): Map[ExprId, NamedExpression] = {
    try {
      val maskEnableResults = plan.output.map { expr =>
        expr -> getAccessResult(table.identifier, expr)
      }.filter(x => isMaskEnabled(x._2))

      val formedMaskers = maskEnableResults.map { case (expr, result) =>
        expr.exprId -> getMasker(expr, result)
      }.filter(_._2 != null).toMap

      val aliasedMaskers = new mutable.HashMap[ExprId, Alias]()

      for (output <- outputs) {
        val newOutput = output transformUp {
          case ar: AttributeReference => formedMaskers.getOrElse(ar.exprId, ar)
        }

        if (!output.equals(newOutput)) {
          val newAlias = newOutput.asInstanceOf[Alias]
          aliasedMaskers.put(output.exprId, newAlias)
        }
      }

      for ((alias, id) <- aliases if formedMaskers.contains(id)) {
        val originalAlias = formedMaskers(id)
        val newChild = originalAlias.child mapChildren {
          case ar: AttributeReference =>
            ar.copy(name = alias.name)(alias.exprId, alias.qualifier)
          case o => o
        }
        val newAlias = Alias(newChild, alias.name)()
        aliasedMaskers.put(alias.exprId, newAlias)
      }

      formedMaskers ++ aliasedMaskers
    } catch {
      case e: Exception => throw e
    }
  }

  private def isMaskEnabled(result: RangerAccessResult): Boolean = {
    result != null && result.isMaskEnabled
  }

  private def hasCatalogTable(plan: LogicalPlan): Boolean = plan match {
    case _: HiveTableRelation => true
    case l: LogicalRelation if l.catalogTable.isDefined => true
    case _ => false
  }

  private def collectAllAliases(plan: LogicalPlan): mutable.HashMap[Alias, ExprId] = {
    val aliases = new mutable.HashMap[Alias, ExprId]()
    plan.transformAllExpressions {
      case a: Alias =>
        a.child transformUp {
          case ne: NamedExpression =>
            aliases.getOrElseUpdate(a, ne.exprId)
            ne
        }
        a
    }
    aliases
  }

  private def collectAllTransformers(
      plan: LogicalPlan,
      aliases: mutable.Map[Alias, ExprId]): Map[ExprId, NamedExpression] = {
    val outputs = plan match {
      case p: Project => p.projectList
      case o => o.output
    }

    plan.collectLeaves().flatMap {
      case h: HiveTableRelation =>
        collectTransformers(h, h.tableMeta, aliases, outputs)
      case l: LogicalRelation if l.catalogTable.isDefined =>
        collectTransformers(l, l.catalogTable.get, aliases, outputs)
      case _ => Seq.empty
    }.toMap
  }

  private def doMasking(plan: LogicalPlan): LogicalPlan = plan match {
    case s: Subquery => s
    case m: SubmarineDataMasking => m // escape the optimize iteration if already masked
    case fixed if fixed.find(_.isInstanceOf[SubmarineDataMasking]).nonEmpty => fixed
    case _ =>
      val aliases = collectAllAliases(plan)
      val transformers = collectAllTransformers(plan, aliases)
      val plansWithTables = plan.collectLeaves().map {
        case h if h.nodeName == "HiveTableRelation" || h.nodeName == "LogicalRelation" =>
          val newPlan = if (transformers.nonEmpty && h.output.exists(o => transformers.contains(o.exprId))) {
            val newOutput = h.output.map(attr => transformers.getOrElse(attr.exprId, attr))
            Project(newOutput, h)
          } else {
            h
          }
          // Call spark analysis here explicitly to resolve UnresolvedFunctions
          val marked = analyzer.execute(newPlan) transformUp {
            case p if hasCatalogTable(p) => SubmarineDataMasking(p)
          }
          // Extract global/local limit if any and apply after masking projection
          val limitExpr: Option[Expression] = plan match {
            case globalLimit: GlobalLimit => Some(globalLimit.limitExpr)
            case localLimit: LocalLimit => Some(localLimit.limitExpr)
            case _ => None
          }

          val markedWithLimit = if (limitExpr.isDefined) Limit(limitExpr.get, marked) else marked

          markedWithLimit transformAllExpressions {
            case s: SubqueryExpression =>
              val SubqueryCompatible(newPlan, _) = SubqueryCompatible(
                SubmarineDataMasking(s.plan), SubqueryExpression.hasCorrelatedSubquery(s))
              s.withNewPlan(newPlan)
          }
          (h, markedWithLimit)
        case _ => null
      }.filter(_ != null).toMap

      plan transformUp {
        case p: HiveTableRelation =>
          if (plansWithTables.contains(p)) {
            plansWithTables(p) transformUp {
              case h: HiveTableRelation =>
                var newAttributeReference = Seq[AttributeReference]()
                h.dataCols.foreach(col => {
                  if (!transformers.contains(col.exprId)) {
                    newAttributeReference :+= col
                  } else {
                    val children = transformers(col.exprId).asInstanceOf[Alias].child.children
                    if (children.isEmpty) {
                      newAttributeReference :+= col
                    } else {
                      val exprId = children.head.asInstanceOf[AttributeReference].exprId
                      newAttributeReference :+= AttributeReference(col.name, col.dataType, col.nullable,
                        col.metadata)(exprId, col.qualifier)
                    }
                  }
                })
                HiveTableRelation(h.tableMeta, newAttributeReference, h.partitionCols)
            }
          } else {
            p
          }
        case p: LogicalRelation =>
          if (plansWithTables.contains(p)) {
            plansWithTables(p) transformUp {
              case l: LogicalRelation =>
                var newOutput = Seq[AttributeReference]()
                l.output.foreach(output => {
                  if (!transformers.contains(output.exprId)) {
                    newOutput :+= output
                  } else {
                    val children = transformers(output.exprId).asInstanceOf[Alias].child.children
                    if (children.isEmpty) {
                      newOutput :+= output
                    } else {
                      val exprId = children.head.asInstanceOf[AttributeReference].exprId
                      newOutput :+= AttributeReference(output.name, output.dataType, output.nullable,
                        output.metadata)(exprId, output.qualifier)
                    }
                  }
                })
                LogicalRelation(l.relation, newOutput, l.catalogTable, l.isStreaming)
            }
          } else {
            p
          }
      }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = plan match {
    case c: Command => c match {
      case c: CreateDataSourceTableAsSelectCommand => c.copy(query = doMasking(c.query))
      case c: CreateHiveTableAsSelectCommand => c.copy(query = doMasking(c.query))
      case c: CreateViewCommand => c.copy(child = doMasking(c.child))
      case i: InsertIntoDataSourceCommand => i.copy(query = doMasking(i.query))
      case i: InsertIntoDataSourceDirCommand => i.copy(query = doMasking(i.query))
      case i: InsertIntoHadoopFsRelationCommand => i.copy(query = doMasking(i.query))
      case i: InsertIntoHiveDirCommand => i.copy(query = doMasking(i.query))
      case i: InsertIntoHiveTable => i.copy(query = doMasking(i.query))
      case s: SaveIntoDataSourceCommand => s.copy(query = doMasking(s.query))
      case cmd => cmd
    }
    case other => doMasking(other)
  }
}
