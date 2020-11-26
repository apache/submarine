package org.apache.submarine.spark.compatible

import org.apache.spark.sql.catalyst.analysis.PersistedView
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, SetCatalogAndNamespace, ShowNamespaces, Subquery}


package object CompatibleCommand {

  type ShowDatabasesCommandCompatible = ShowNamespaces
  type SetDatabaseCommandCompatible = SetCatalogAndNamespace
}

object PersistedViewCompatible {
  val obj: PersistedView.type = PersistedView
}


