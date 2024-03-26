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
import io.fabric8.kubernetes.api.model.KubernetesResource;

import java.util.Objects;

@JsonDeserialize(
        using = JsonDeserializer.None.class
)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"lastProbeTime", "lastTransitionTime", "message", "reason", "status", "type"})
@JsonIgnoreProperties(ignoreUnknown = true)
public class NotebookCondition implements KubernetesResource {

  private String lastProbeTime;

  // Add from submarine 0.8.0
  // From Notebook Controller 1.6.0, `NotebookCondition` add a new time variable `lastTransitionTime`
  // https://github.com/kubeflow/kubeflow/blob/v1.6.0/components/notebook-controller/api/v1/notebook_types.go#L53
  // When the lastProbeTime times are the same, we will follow this variable again for comparison.
  private String lastTransitionTime;

  private String message;

  private String reason;

  private String status;

  private String type;

  public String getLastProbeTime() {
    return lastProbeTime;
  }

  public void setLastProbeTime(String lastProbeTime) {
    this.lastProbeTime = lastProbeTime;
  }

  public String getLastTransitionTime() {
    return lastTransitionTime;
  }

  public void setLastTransitionTime(String lastTransitionTime) {
    this.lastTransitionTime = lastTransitionTime;
  }

  public String getMessage() {
    return message;
  }

  public void setMessage(String message) {
    this.message = message;
  }

  public String getReason() {
    return reason;
  }

  public void setReason(String reason) {
    this.reason = reason;
  }

  public String getStatus() {
    return status;
  }

  public void setStatus(String status) {
    this.status = status;
  }

  public String getType() {
    return type;
  }

  public void setType(String type) {
    this.type = type;
  }

  @Override
  public String toString() {
    return "NotebookCondition{" +
            "lastProbeTime='" + lastProbeTime + '\'' +
            ", lastTransitionTime='" + lastTransitionTime + '\'' +
            ", message='" + message + '\'' +
            ", reason='" + reason + '\'' +
            ", status='" + status + '\'' +
            ", type='" + type + '\'' +
            '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    NotebookCondition condition = (NotebookCondition) o;
    return Objects.equals(lastProbeTime, condition.lastProbeTime)
            && Objects.equals(lastTransitionTime, condition.lastTransitionTime)
            && Objects.equals(message, condition.message)
            && Objects.equals(reason, condition.reason)
            && Objects.equals(status, condition.status)
            && Objects.equals(type, condition.type);
  }

  @Override
  public int hashCode() {
    return Objects.hash(lastProbeTime, lastTransitionTime, message, reason, status, type);
  }
}
