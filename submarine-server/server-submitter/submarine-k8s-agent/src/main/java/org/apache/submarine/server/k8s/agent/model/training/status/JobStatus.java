package org.apache.submarine.server.k8s.agent.model.training.status;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import io.fabric8.kubernetes.api.model.KubernetesResource;

import java.util.List;
import java.util.Map;
import java.util.Objects;

@JsonDeserialize(
        using = JsonDeserializer.None.class
)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"apiVersion", "kind", "metadata", "completionTime", "conditions", "lastReconcileTime", "replicaStatuses", "startTime"})
@JsonIgnoreProperties(ignoreUnknown = true)
public class JobStatus implements KubernetesResource {

    private String completionTime;

    private List<JobCondition> conditions;

    private String lastReconcileTime;

    private Map<String, ReplicaStatus> replicaStatuses;

    private String startTime;

    public String getCompletionTime() {
        return completionTime;
    }

    public void setCompletionTime(String completionTime) {
        this.completionTime = completionTime;
    }

    public List<JobCondition> getConditions() {
        return conditions;
    }

    public void setConditions(List<JobCondition> conditions) {
        this.conditions = conditions;
    }

    public String getLastReconcileTime() {
        return lastReconcileTime;
    }

    public void setLastReconcileTime(String lastReconcileTime) {
        this.lastReconcileTime = lastReconcileTime;
    }

    public Map<String, ReplicaStatus> getReplicaStatuses() {
        return replicaStatuses;
    }

    public void setReplicaStatuses(Map<String, ReplicaStatus> replicaStatuses) {
        this.replicaStatuses = replicaStatuses;
    }

    public String getStartTime() {
        return startTime;
    }

    public void setStartTime(String startTime) {
        this.startTime = startTime;
    }

    @Override
    public String toString() {
        return "JobStatus{" +
                "completionTime='" + completionTime + '\'' +
                ", conditions=" + conditions +
                ", lastReconcileTime='" + lastReconcileTime + '\'' +
                ", replicaStatuses=" + replicaStatuses +
                ", startTime='" + startTime + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JobStatus jobStatus = (JobStatus) o;
        return Objects.equals(completionTime, jobStatus.completionTime)
                && Objects.equals(conditions, jobStatus.conditions)
                && Objects.equals(lastReconcileTime, jobStatus.lastReconcileTime)
                && Objects.equals(replicaStatuses, jobStatus.replicaStatuses)
                && Objects.equals(startTime, jobStatus.startTime);
    }

    @Override
    public int hashCode() {
        return Objects.hash(completionTime, conditions, lastReconcileTime, replicaStatuses, startTime);
    }
}
