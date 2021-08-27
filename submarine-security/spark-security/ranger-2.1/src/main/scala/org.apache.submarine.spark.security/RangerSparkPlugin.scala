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

package org.apache.submarine.spark.security

import java.io.{File, IOException}
import org.apache.commons.logging.LogFactory
import org.apache.ranger.authorization.hadoop.config.RangerPluginConfig
import org.apache.ranger.plugin.service.RangerBasePlugin

object RangerSparkPlugin extends RangerBasePlugin("spark", "sparkSql") {

  private val LOG = LogFactory.getLog(RangerSparkPlugin.getClass)

  private val rangerConf: RangerPluginConfig = this.getConfig
  val showColumnsOption: String = rangerConf.get(
    "xasecure.spark.describetable.showcolumns.authorization.option", "NONE")

  lazy val fsScheme: Array[String] = rangerConf
    .get("ranger.plugin.spark.urlauth.filesystem.schemes", "hdfs:,file:")
    .split(",")
    .map(_.trim)

  override def init(): Unit = {
    super.init()
    val cacheDir = new File(rangerConf.get("ranger.plugin.spark.policy.cache.dir"))
    if (cacheDir.exists() &&
      (!cacheDir.isDirectory || !cacheDir.canRead || !cacheDir.canWrite)) {
      throw new IOException("Policy cache directory already exists at" +
        cacheDir.getAbsolutePath + ", but it is unavailable")
    }

    if (!cacheDir.exists() && !cacheDir.mkdirs()) {
      throw new IOException("Unable to create ranger policy cache directory at" +
        cacheDir.getAbsolutePath)
    }
    LOG.info("Policy cache directory successfully set to " + cacheDir.getAbsolutePath)
  }

  init()
}
