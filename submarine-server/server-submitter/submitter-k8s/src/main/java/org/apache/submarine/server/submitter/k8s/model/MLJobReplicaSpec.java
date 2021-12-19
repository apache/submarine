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

import com.google.gson.annotations.SerializedName;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;

import java.math.BigDecimal;

/**
 * Both PyTorch and Tensorflow CRD definition uses this.
 * */
public class MLJobReplicaSpec {
  @SerializedName("replicas")
  private Integer replicas;

  @SerializedName("template")
  private V1PodTemplateSpec template;

  /**
   * Always, OnFailure, ExitCode, Never
   */
  @SerializedName("restartPolicy")
  private String restartPolicy = "OnFailure";

  public MLJobReplicaSpec() {}

  /**
   * Number of desired pod.
   * @return number
   */
  public Integer getReplicas() {
    return replicas;
  }

  /**
   * Set the number of desired pod
   * @param replicas number
   */
  public void setReplicas(Integer replicas) {
    this.replicas = replicas;
  }

  /**
   * Get the pod template
   * @return pod template spec
   */
  public V1PodTemplateSpec getTemplate() {
    return template;
  }

  /**
   * Set the pod template
   * @param template pod template
   */
  public void setTemplate(V1PodTemplateSpec template) {
    this.template = template;
  }

  /**
   * Get the restart policy.
   * Default is OnFailure.
   * Supports: Always, OnFailure, ExitCode, Never
   * @return policy name
   */
  public String getRestartPolicy() {
    return restartPolicy;
  }

  /**
   * Set the restart policy.
   * @param restartPolicy policy name, the range is [Always, OnFailure, ExitCode, Never]
   */
  public void setRestartPolicy(String restartPolicy) {
    this.restartPolicy = restartPolicy;
  }

  public String getContainerCommand() {
    V1PodTemplateSpec podSpec = getTemplate();
    return String.join(" ",
        podSpec.getSpec().getContainers().get(0).getCommand());
  }

  public String getContainerMemMB() {
    V1PodTemplateSpec podSpec = getTemplate();
    return String.join(" ",
        podSpec.getSpec().getContainers().get(0)
            .getResources().getLimits().get("memory").
            getNumber().divide(BigDecimal.valueOf(1000000)).toString() + "M");
  }

  public String getContainerCpu() {
    V1PodTemplateSpec podSpec = getTemplate();
    return podSpec.getSpec().getContainers().get(0)
        .getResources().getLimits().get("cpu").getNumber().toString();
  }

  public String getContainerImageName() {
    V1PodTemplateSpec podSpec = getTemplate();
    return podSpec.getSpec().getContainers().get(0)
        .getImage();
  }

}
