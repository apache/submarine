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

package org.apache.submarine.spark.security.api

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.optimizer.{SubmarineConfigurationCheckExtension, SubmarineDataMaskingExtension, SubmarinePushPredicatesThroughExtensions, SubmarineRowFilterExtension, SubmarineSparkRangerAuthorizationExtension}
import org.apache.spark.sql.execution.SubmarineSparkPlanOmitStrategy

import org.apache.submarine.spark.security.Extensions

/**
 * ACL Management for Apache Spark SQL with Apache Ranger, enabling:
 * <ul>
 *   <li>Table/Column level authorization</li>
 *   <li>Row level filtering</li>
 *   <li>Data masking</li>
 * <ul>
 *
 * To work with Spark SQL, we need to enable it via spark extensions
 *
 * spark.sql.extensions=org.apache.submarine.spark.security.api.RangerSparkSQLExtension
 */
class RangerSparkSQLExtension extends Extensions {
  override def apply(ext: SparkSessionExtensions): Unit = {
    ext.injectCheckRule(SubmarineConfigurationCheckExtension)
    ext.injectOptimizerRule(SubmarineSparkRangerAuthorizationExtension)
    ext.injectOptimizerRule(SubmarineRowFilterExtension)
    ext.injectOptimizerRule(SubmarineDataMaskingExtension)
    ext.injectOptimizerRule(SubmarinePushPredicatesThroughExtensions)
    ext.injectPlannerStrategy(SubmarineSparkPlanOmitStrategy)
  }
}
