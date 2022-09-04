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
package org.apache.submarine.serve.seldon;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.gson.annotations.SerializedName;
import io.kubernetes.client.common.KubernetesObject;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1ObjectMetaBuilder;
import org.apache.submarine.serve.utils.SeldonConstants;
import org.apache.submarine.server.k8s.utils.K8sUtils;

public class SeldonDeployment implements KubernetesObject {

  public static final String MODEL_NAME_LABEL = "model-name";
  public static final String MODEL_ID_LABEL = "model-id";
  public static final String MODEL_VERSION_LABEL = "model-version";

  @SerializedName("apiVersion")
  private String apiVersion = SeldonConstants.API_VERSION;

  @SerializedName("kind")
  private String kind = SeldonConstants.KIND;

  @SerializedName("metadata")
  private V1ObjectMeta metadata;

  @SerializedName("spec")
  private SeldonDeploymentSpec spec;

  @JsonIgnore
  private Long id;

  @JsonIgnore
  private String resourceName;

  @JsonIgnore
  private String modelName;

  @JsonIgnore
  private String modelURI;

  @JsonIgnore
  private Integer modelVersion;

  @JsonIgnore
  private String modelId;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public String getResourceName() {
    return resourceName;
  }

  public void setResourceName(String resourceName) {
    this.resourceName = resourceName;
  }

  public String getModelName() {
    return modelName;
  }

  public void setModelName(String modelName) {
    this.modelName = modelName;
  }

  public String getModelURI() {
    return modelURI;
  }

  public void setModelURI(String modelURI) {
    this.modelURI = modelURI;
  }

  public String getModelId() {
    return modelId;
  }

  public void setModelId(String modelId) {
    this.modelId = modelId;
  }

  public Integer getModelVersion() {
    return modelVersion;
  }

  public void setModelVersion(Integer modelVersion) {
    this.modelVersion = modelVersion;
  }

  public SeldonDeployment() {
  }

  public SeldonDeployment(Long id, String resourceName, String modelName, Integer modelVersion,
                          String modelId, String modelURI) {
    this.id = id;
    this.resourceName = resourceName;
    this.modelName = modelName;
    this.modelVersion = modelVersion;
    this.modelId = modelId;
    this.modelURI = modelURI;

    V1ObjectMetaBuilder metaBuilder = new V1ObjectMetaBuilder();
    metaBuilder.withNamespace(K8sUtils.getNamespace())
            .withName(resourceName)
            .addToLabels(MODEL_NAME_LABEL, modelName)
            .addToLabels(MODEL_ID_LABEL, modelId)
            .addToLabels(MODEL_VERSION_LABEL, String.valueOf(modelVersion));
    setMetadata(metaBuilder.build());
  }

  // transient to avoid being serialized
  private transient String group = SeldonConstants.GROUP;

  private transient String version = SeldonConstants.VERSION;

  private transient String plural = SeldonConstants.PLURAL;

  public String getApiVersion() {
    return apiVersion;
  }

  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public String getKind() {
    return kind;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  public V1ObjectMeta getMetadata() {
    return metadata;
  }

  public void setMetadata(V1ObjectMeta metadata) {
    this.metadata = metadata;
  }

  public String getGroup() {
    return group;
  }

  public void setGroup(String group) {
    this.group = group;
  }

  public String getVersion() {
    return version;
  }

  public void setVersion(String version) {
    this.version = version;
  }

  public String getPlural() {
    return plural;
  }

  public void setPlural(String plural) {
    this.plural = plural;
  }

  public void setSpec(SeldonDeploymentSpec seldonDeploymentSpec){
    this.spec = seldonDeploymentSpec;
  }

  public SeldonDeploymentSpec getSpec() {
    return spec;
  }

  public void addPredictor(SeldonPredictor seldonPredictor) {
    this.spec.addPredictor(seldonPredictor);
  }
}
