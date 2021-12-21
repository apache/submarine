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
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;

import java.math.BigDecimal;
import java.util.List;

public class NotebookCRSpec {

  public NotebookCRSpec() {

  }

  @SerializedName("template")
  private V1PodTemplateSpec template;

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
   * Get memory resource for container
   * @return memory in Gi
   */
  public String getContainerMemory() {
    V1PodTemplateSpec podSpec = getTemplate();
    return String.join(" ",
            podSpec.getSpec().getContainers().get(0)
                    .getResources().getLimits().get("memory").
                    getNumber().divide(BigDecimal.valueOf(1024 * 1024 * 1024)).toString() + "Gi");
  }

  /**
   * Get CPU resource for container
   * @return CPU in VCores
   */
  public String getContainerCpu() {
    V1PodTemplateSpec podSpec = getTemplate();
    return podSpec.getSpec().getContainers().get(0)
            .getResources().getLimits().get("cpu").getNumber().toString();
  }

  /**
   * Get GPU resource for container
   * @return GPU
   */
  public String getContainerGpu() {
    V1PodTemplateSpec podSpec = getTemplate();
    return podSpec.getSpec().getContainers().get(0)
            .getResources().getLimits().get("nvidia.com/gpu").getNumber().toString();
  }

  /**
   * Get the image name
   * @return image name
   */
  public String getContainerImageName() {
    V1PodTemplateSpec podSpec = getTemplate();
    return podSpec.getSpec().getContainers().get(0).getImage();
  }

  public List<V1EnvVar> getEnvs() {
    V1PodTemplateSpec podSpec = getTemplate();
    return podSpec.getSpec().getContainers().get(0).getEnv();
  }
}
