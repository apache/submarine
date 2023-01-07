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
@JsonPropertyOrder({"apiVersion", "kind", "metadata", "active", "failed", "succeeded"})
@JsonIgnoreProperties(ignoreUnknown = true)
public class ReplicaStatus implements KubernetesResource {

    private Integer active;

    private Integer failed;

    private Integer succeeded;

    public Integer getActive() {
        return active;
    }

    public void setActive(Integer active) {
        this.active = active;
    }

    public Integer getFailed() {
        return failed;
    }

    public void setFailed(Integer failed) {
        this.failed = failed;
    }

    public Integer getSucceeded() {
        return succeeded;
    }

    public void setSucceeded(Integer succeeded) {
        this.succeeded = succeeded;
    }

    @Override
    public String toString() {
        return "ReplicaStatus{" +
                "active=" + active +
                ", failed=" + failed +
                ", succeeded=" + succeeded +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ReplicaStatus that = (ReplicaStatus) o;
        return Objects.equals(active, that.active)
                && Objects.equals(failed, that.failed)
                && Objects.equals(succeeded, that.succeeded);
    }

    @Override
    public int hashCode() {
        return Objects.hash(active, failed, succeeded);
    }
}
