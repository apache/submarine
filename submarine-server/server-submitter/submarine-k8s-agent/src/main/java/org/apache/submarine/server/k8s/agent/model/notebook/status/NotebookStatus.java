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

package org.apache.submarine.server.k8s.agent.model.notebook.status;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import io.fabric8.kubernetes.api.model.ContainerState;
import io.fabric8.kubernetes.api.model.KubernetesResource;

import java.util.List;
import java.util.Objects;

@JsonDeserialize(
        using = JsonDeserializer.None.class
)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"conditions", "readyReplicas", "containerState"})
@JsonIgnoreProperties(ignoreUnknown = true)
public class NotebookStatus implements KubernetesResource {

  private List<NotebookCondition> conditions;

  private Integer readyReplicas;

  private ContainerState containerState;

  public List<NotebookCondition> getConditions() {
    return conditions;
  }

  public void setConditions(List<NotebookCondition> conditions) {
    this.conditions = conditions;
  }

  public Integer getReadyReplicas() {
    return readyReplicas;
  }

  public void setReadyReplicas(Integer readyReplicas) {
    this.readyReplicas = readyReplicas;
  }

  public ContainerState getContainerState() {
    return containerState;
  }

  public void setContainerState(ContainerState containerState) {
    this.containerState = containerState;
  }

  @Override
  public String toString() {
    return "NotebookStatus{" +
            "conditions=" + conditions +
            ", readyReplicas=" + readyReplicas +
            ", containerState=" + containerState +
            '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    NotebookStatus that = (NotebookStatus) o;
    return Objects.equals(conditions, that.conditions)
            && Objects.equals(readyReplicas, that.readyReplicas)
            && Objects.equals(containerState, that.containerState);
  }

  @Override
  public int hashCode() {
    return Objects.hash(conditions, readyReplicas, containerState);
  }
}
