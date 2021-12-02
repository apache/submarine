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

import com.google.gson.annotations.SerializedName;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.serve.utils.SeldonConstants;

import java.util.ArrayList;
import java.util.List;

public class SeldonDeployment {
  @SerializedName("apiVersion")
  private String apiVersion = SeldonConstants.API_VERSION;

  @SerializedName("kind")
  private String kind = SeldonConstants.KIND;

  @SerializedName("metadata")
  private V1ObjectMeta metadata;

  @SerializedName("spec")
  private SeldonDeploymentSpec spec;

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

  public void addPredictor(SeldonPredictor seldonPredictor) {
    this.spec.addPredictor(seldonPredictor);
  }

  public static class SeldonDeploymentSpec {
    public SeldonDeploymentSpec(String protocol) {
      setProtocol(protocol);
    }

    @SerializedName("protocol")
    private String protocol;
    @SerializedName("predictors")
    private List<SeldonPredictor> predictors = new ArrayList<>();

    public String getProtocol() {
      return protocol;
    }

    public void setProtocol(String protocol) {
      this.protocol = protocol;
    }

    public void addPredictor(SeldonPredictor seldonPredictor) {
      predictors.add(seldonPredictor);
    }
  }
}
