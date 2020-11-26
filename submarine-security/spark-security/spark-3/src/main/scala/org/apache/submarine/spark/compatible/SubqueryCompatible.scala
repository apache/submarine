package org.apache.submarine.spark.compatible

import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Subquery}


case class SubqueryCompatible(child: LogicalPlan)

object SubqueryCompatible {
  def apply(child: LogicalPlan): Subquery = Subquery(child, false)

  def unapply(child: LogicalPlan): Option[LogicalPlan] = Option(Subquery.unapply(SubqueryCompatible(child)).get._1)
}


