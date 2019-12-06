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

package org.apache.submarine.server.submitter.k8s.model;

import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;

import java.util.Map;

/**
 * The response job for CRD API.
 *   POST:   /apis/{group}/{version}/namespaces/{namespace}/{plural}
 *   GET:    /apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}
 *   DELETE: /apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}
 */
public class CustomResourceJob {
  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  protected String kind;

  @SerializedName("metadata")
  private ObjectMeta metadata;

  @SerializedName("spec")
  private Map<String, Object> spec;

  public String getApiVersion() {
    return apiVersion;
  }

  public String getKind() {
    return kind;
  }

  public ObjectMeta getMetadata() {
    return metadata;
  }

  public Map<String, Object> getSpec() {
    return spec;
  }

  @Override
  public String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(this);
  }
}
