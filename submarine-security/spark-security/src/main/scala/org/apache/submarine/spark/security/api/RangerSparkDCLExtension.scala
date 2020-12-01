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

import org.apache.submarine.spark.security.Extensions
import org.apache.submarine.spark.security.parser.SubmarineSqlParser

/**
 * An extension for Spark SQL to activate DCL(Data Control Language)
 *
 * Scala example to create a `SparkSession` with the Submarine DCL parser::
 * {{{
 *    import org.apache.spark.sql.SparkSession
 *
 *    val spark = SparkSession
 *       .builder()
 *       .appName("...")
 *       .master("...")
 *       .config("spark.sql.extensions",
 *         "org.apache.submarine.spark.security.api.RangerSparkDCLExtension")
 *       .getOrCreate()
 * }}}
 *
 * Java example to create a `SparkSession` with the Submarine DCL parser:
 * {{{
 *    import org.apache.spark.sql.SparkSession;
 *
 *    SparkSession spark = SparkSession
 *                 .builder()
 *                 .appName("...")
 *                 .master("...")
 *                 .config("spark.sql.extensions",
 *                     "org.apache.submarine.spark.security.api.RangerSparkDCLExtension")
 *                 .getOrCreate();
 * }}}
 *
 * @since 0.4.0
 */
class RangerSparkDCLExtension extends Extensions {
  override def apply(ext: SparkSessionExtensions): Unit = {
    ext.injectParser((_, parser) => new SubmarineSqlParser(parser))
  }
}
