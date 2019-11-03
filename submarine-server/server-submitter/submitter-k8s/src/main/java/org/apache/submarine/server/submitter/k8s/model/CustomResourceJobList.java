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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The response for CRD API.
 *   GET: /apis/{group}/{version}/namespaces/{namespace}/{plural}
 */
public class CustomResourceJobList {
  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("items")
  private List<Object> items = new ArrayList<>();

  @SerializedName("kind")
  protected String kind;

  @SerializedName("metadata")
  private ListMeta metadata;

  public String getApiVersion() {
    return apiVersion;
  }

  public List<Object> getItems() {
    return items;
  }

  public String getKind() {
    return kind;
  }

  public ListMeta getMetadata() {
    return metadata;
  }

  @Override
  public int hashCode() {
    return Objects.hash(apiVersion, items, kind, metadata);
  }

  @Override
  public boolean equals(java.lang.Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CustomResourceJobList jobList = (CustomResourceJobList) o;
    return Objects.equals(this.apiVersion, jobList.apiVersion) &&
        Objects.equals(this.items, jobList.items) &&
        Objects.equals(this.kind, jobList.kind) &&
        Objects.equals(this.metadata, jobList.metadata);
  }

  @Override
  public String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(this);
  }
}
