package org.apache.submarine.spark.compatible

import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Subquery}


class SubqueryCompatible(override val child: LogicalPlan) extends Subquery(child: LogicalPlan){

  def unapply(arg: Subquery): Option[LogicalPlan] = Subquery.unapply(arg)

}

object SubqueryCompatible {

  def apply( child: LogicalPlan): Subquery = Subquery(child)

  def unapply(arg: Subquery): Option[LogicalPlan] = Subquery.unapply(arg)
}



