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

import java.nio.file.{FileSystems, Files}
import java.util
import com.google.gson.GsonBuilder
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.ranger.admin.client.RangerAdminClient
import org.apache.ranger.plugin.model.RangerRole
import org.apache.ranger.plugin.util.{GrantRevokeRequest, GrantRevokeRoleRequest, RangerRoles, RangerUserStore, ServicePolicies, ServiceTags}

class RangerAdminClientImpl extends RangerAdminClient {
  private val LOG: Log = LogFactory.getLog(classOf[RangerAdminClientImpl])
  private val cacheFilename = "sparkSql_hive_jenkins.json"
  private val gson =
    new GsonBuilder().setDateFormat("yyyyMMdd-HH:mm:ss.SSS-Z").setPrettyPrinting().create
  private var policies: ServicePolicies = _

  override def init(serviceName: String, appId: String, configPropertyPrefix: String, var4: Configuration): Unit = {
    if (policies == null) {
      val basedir = this.getClass.getProtectionDomain.getCodeSource.getLocation.getPath
      val cachePath = FileSystems.getDefault.getPath(basedir, cacheFilename)
      LOG.info("Reading policies from " + cachePath)
      val bytes = Files.readAllBytes(cachePath)
      policies = gson.fromJson(new String(bytes), classOf[ServicePolicies])
    }
  }

  override def getServicePoliciesIfUpdated(lastKnownVersion: Long, lastActivationTimeInMillis: Long): ServicePolicies = {
    policies
  }

  override def grantAccess(request: GrantRevokeRequest): Unit = {}

  override def revokeAccess(request: GrantRevokeRequest): Unit = {}

  override def getServiceTagsIfUpdated(lastKnownVersion: Long, lastActivationTimeInMillis: Long): ServiceTags = null

  override def getTagTypes(tagTypePattern: String): util.List[String] = null

  override def getRolesIfUpdated(l: Long, l1: Long): RangerRoles = null

  override def createRole(rangerRole: RangerRole): RangerRole = null

  override def dropRole(s: String, s1: String): Unit = {}

  override def getAllRoles(s: String): util.List[String] = null

  override def getUserRoles(s: String): util.List[String] = null

  override def getRole(s: String, s1: String): RangerRole = null

  override def grantRole(grantRevokeRoleRequest: GrantRevokeRoleRequest): Unit = {}

  override def revokeRole(grantRevokeRoleRequest: GrantRevokeRoleRequest): Unit = {}

  override def getUserStoreIfUpdated(l: Long, l1: Long): RangerUserStore = null
}
