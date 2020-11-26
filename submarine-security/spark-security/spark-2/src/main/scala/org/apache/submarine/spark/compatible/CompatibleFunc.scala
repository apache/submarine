package org.apache.submarine.spark.compatible

import org.apache.spark.sql.catalyst.analysis.UnresolvedRelation
import org.apache.spark.sql.execution.command.{AnalyzeColumnCommand, SetDatabaseCommand, ShowDatabasesCommand}

object CompatibleFunc {

  def getPattern(child: ShowDatabasesCommand) = child.databasePattern

  def getCatLogName(s: SetDatabaseCommand) = s.databaseName

  def analyzeColumnName(column: AnalyzeColumnCommand) = column.columnNames

  def tableIdentifier(u: UnresolvedRelation) = u.tableIdentifier
}
