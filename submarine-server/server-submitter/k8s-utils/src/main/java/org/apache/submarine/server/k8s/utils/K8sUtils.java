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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;

/**
 * Utility methods for common k8s operations.
 *
 * @author Chi-Sheng Liu
 * @since 0.8.0-SNAPSHOT
 */
public abstract class K8sUtils {

  private static final Logger LOG = LoggerFactory.getLogger(K8sUtils.class);
  private static String namespace = null;

  /**
   * Get the current Kubernetes namespace.
   * @return The current Kubernetes namespace.
   */
  public static String getNamespace() {
    if (namespace == null) {
      namespace = System.getenv("ENV_NAMESPACE");
      if (namespace == null) {
        namespace = "default";
      }
      LOG.info("Namespace: {}", namespace);
    }
    return namespace;
  }

  public static final DateTimeFormatter UTC_DATE_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssX");

  /**
   * In k8s resource time declarations, the usual return is something like this:
   * <p>
   * creationTimestamp: '2023-05-23T09:01:12Z'
   * <p>
   * So we try to do the same for our return datetime format
   */
  public static String castOffsetDatetimeToString(OffsetDateTime odt) {
    if (odt == null) return null;
    return UTC_DATE_FORMATTER.format(odt.withOffsetSameInstant(ZoneOffset.UTC));
  }
}
