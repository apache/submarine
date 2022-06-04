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

package org.apache.submarine.server.submitter.k8s.model.ingressroute;

import com.google.gson.annotations.SerializedName;

import java.util.Map;
import java.util.Set;

@Deprecated
public class SpecRoute {

  public SpecRoute() {

  }

  @SerializedName("match")
  private String match;

  @SerializedName("kind")
  private String kind;

  @SerializedName("services")
  private Set<Map<String, Object>> services;

  @SerializedName("middlewares")
  private Set<Map<String, String>> middlewares;

  public String getMatch() {
    return match;
  }

  public void setMatch(String match) {
    this.match = match;
  }

  public String getKind() {
    return kind;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  public Set<Map<String, Object>> getServices() {
    return services;
  }

  public void setServices(Set<Map<String, Object>> services) {
    this.services = services;
  }

  public Set<Map<String, String>> getMiddlewares() {
    return middlewares;
  }

  public void setMiddlewares(Set<Map<String, String>> middlewares) {
    this.middlewares = middlewares;
  }
}
