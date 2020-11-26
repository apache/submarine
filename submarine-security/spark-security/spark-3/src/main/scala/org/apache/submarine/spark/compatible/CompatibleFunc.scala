package org.apache.submarine.spark.compatible

import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedRelation
import org.apache.spark.sql.catalyst.plans.logical.{SetCatalogAndNamespace, ShowNamespaces}
import org.apache.spark.sql.execution.command.AnalyzeColumnCommand

object CompatibleFunc {

  def getPattern(child: ShowNamespaces) = child.pattern

  def getCatLogName(s: SetCatalogAndNamespace) = s.catalogName

  def analyzeColumnName(column: AnalyzeColumnCommand) = column.columnNames.get

  def tableIdentifier(u: UnresolvedRelation) = TableIdentifier.apply(u.tableName)
}
