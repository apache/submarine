package org.apache.submarine.spark.compatible

import org.apache.spark.sql.execution.command.{PersistedView, SetDatabaseCommand, ShowDatabasesCommand}

package object CompatibleCommand {

  type ShowDatabasesCommandCompatible = ShowDatabasesCommand
  type SetDatabaseCommandCompatible = SetDatabaseCommand

}

object PersistedViewCompatible {
  val obj: PersistedView.type = PersistedView
}
