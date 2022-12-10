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
import org.apache.submarine.serve.utils.SeldonConstants;

/**
 * Seldon graph
 * <p>
 * Define the graph of every predictor in `spec.predictors[*].graph`, for more we can see:
 * <a href="https://docs.seldon.io/projects/seldon-core/en/latest/graph/inference-graph.html">
 *   Inference Graph
 * </a>
 */
public class SeldonGraph {

  /**
   * Graph name, we generally order by version, e.g.
   * version-1, version-2, version-3 ...
   */
  @SerializedName("name")
  private String name;

  /**
   * Graph implementation, can be:
   * TENSORFLOW_SERVER, TRITON_SERVER or XGBOOST_SERVER
   */
  @SerializedName("implementation")
  private String implementation;

  /**
   * Model storage path on S3(minio), e.g.
   * s3://submarine/registry/${model_version_path}/${model_name}
   */
  @SerializedName("modelUri")
  private String modelUri;

  /**
   * S3(minio) secret, We have created `Secret` resource by default when creating the submarine
   */
  @SerializedName("envSecretRefName")
  private String envSecretRefName = SeldonConstants.ENV_SECRET_REF_NAME;

  public SeldonGraph() {
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public String getImplementation() {
    return implementation;
  }

  public void setImplementation(String implementation) {
    this.implementation = implementation;
  }

  public String getModelUri() {
    return modelUri;
  }

  public void setModelUri(String modelUri) {
    this.modelUri = modelUri;
  }

  public String getEnvSecretRefName() {
    return envSecretRefName;
  }

  public void setEnvSecretRefName(String envSecretRefName) {
    this.envSecretRefName = envSecretRefName;
  }
}
