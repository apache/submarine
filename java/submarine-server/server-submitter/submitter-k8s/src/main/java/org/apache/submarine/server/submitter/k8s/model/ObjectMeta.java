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

import java.util.List;
import java.util.Map;

/**
 * ObjectMeta is metadata that all persisted resources must have, which includes all objects users
 * must create.
 */
public class ObjectMeta {
  @SerializedName("annotations")
  private Map<String, String> annotations;

  @SerializedName("clusterName")
  private String clusterName;

  @SerializedName("creationTimestamp")
  private String creationTimestamp;

  @SerializedName("deletionGracePeriodSeconds")
  private Long deletionGracePeriodSeconds;

  @SerializedName("deletionTimestamp")
  private String deletionTimestamp;

  @SerializedName("finalizers")
  private List<String> finalizers;

  @SerializedName("generateName")
  private String generateName;

  @SerializedName("generation")
  private Long generation;

  @SerializedName("labels")
  private Map<String, String> labels = null;

  @SerializedName("name")
  private String name = null;

  @SerializedName("namespace")
  private String namespace = null;

  @SerializedName("resourceVersion")
  private String resourceVersion = null;

  @SerializedName("selfLink")
  private String selfLink = null;

  @SerializedName("uid")
  private String uid = null;

  public Map<String, String> getAnnotations() {
    return annotations;
  }

  public String getClusterName() {
    return clusterName;
  }

  public String getCreationTimestamp() {
    return creationTimestamp;
  }

  public Long getDeletionGracePeriodSeconds() {
    return deletionGracePeriodSeconds;
  }

  public String getDeletionTimestamp() {
    return deletionTimestamp;
  }

  public List<String> getFinalizers() {
    return finalizers;
  }

  public String getGenerateName() {
    return generateName;
  }

  public Long getGeneration() {
    return generation;
  }

  public Map<String, String> getLabels() {
    return labels;
  }

  public String getName() {
    return name;
  }

  public String getNamespace() {
    return namespace;
  }

  public String getResourceVersion() {
    return resourceVersion;
  }

  public String getSelfLink() {
    return selfLink;
  }

  public String getUid() {
    return uid;
  }

  @Override
  public String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(this);
  }
}
