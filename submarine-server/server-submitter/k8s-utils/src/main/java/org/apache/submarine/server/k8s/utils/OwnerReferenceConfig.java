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

package org.apache.submarine.server.k8s.utils;

/**
 * OwnerReference config
 * We currently get the configuration by environment variables
 */
public class OwnerReferenceConfig {

  public static final String SUBMARINE_APIVERSION = "SUBMARINE_APIVERSION";
  public static final String SUBMARINE_KIND = "SUBMARINE_KIND";
  public static final String SUBMARINE_NAME = "SUBMARINE_NAME";
  public static final String SUBMARINE_UID = "SUBMARINE_UID";

  public static final String DEFAULT_SUBMARINE_APIVERSION = "submarine.apache.org/v1alpha1";
  public static final String DEFAULT_SUBMARINE_KIND = "Submarine";

  /**
   * Get submarine apiVersion
   */
  public static String getSubmarineApiversion() {
    String apiVersion = System.getenv(SUBMARINE_APIVERSION);
    return apiVersion == null || apiVersion.isEmpty() ? DEFAULT_SUBMARINE_APIVERSION : apiVersion;
  }

  /**
   * Get submarine kind
   */
  public static String getSubmarineKind() {
    String kind = System.getenv(SUBMARINE_KIND);
    return kind == null || kind.isEmpty() ? DEFAULT_SUBMARINE_KIND : kind;
  }

  /**
   * Get submarine CR name
   */
  public static String getSubmarineName() {
    return System.getenv(SUBMARINE_NAME);
  }

  /**
   * Get submarine owner references uid
   */
  public static String getSubmarineUid() {
    return System.getenv(SUBMARINE_UID);
  }

}
