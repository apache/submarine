package org.apache.submarine.server.k8s.agent.model.training.status;

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
@JsonPropertyOrder({"lastTransitionTime", "lastUpdateTime", "message", "reason", "status", "type"})
@JsonIgnoreProperties(ignoreUnknown = true)
public class JobCondition implements KubernetesResource {

  private String lastTransitionTime;

  private String lastUpdateTime;

  private String message;

  private String reason;

  private String status;

  private String type;

  public String getLastTransitionTime() {
    return lastTransitionTime;
  }

  public void setLastTransitionTime(String lastTransitionTime) {
    this.lastTransitionTime = lastTransitionTime;
  }

  public String getLastUpdateTime() {
    return lastUpdateTime;
  }

  public void setLastUpdateTime(String lastUpdateTime) {
    this.lastUpdateTime = lastUpdateTime;
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
    return "JobCondition{" +
            "lastTransitionTime='" + lastTransitionTime + '\'' +
            ", lastUpdateTime='" + lastUpdateTime + '\'' +
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
    JobCondition that = (JobCondition) o;
    return Objects.equals(lastTransitionTime, that.lastTransitionTime)
            && Objects.equals(lastUpdateTime, that.lastUpdateTime)
            && Objects.equals(message, that.message)
            && Objects.equals(reason, that.reason)
            && Objects.equals(status, that.status)
            && Objects.equals(type, that.type);
  }

  @Override
  public int hashCode() {
    return Objects.hash(lastTransitionTime, lastUpdateTime, message, reason, status, type);
  }
}
