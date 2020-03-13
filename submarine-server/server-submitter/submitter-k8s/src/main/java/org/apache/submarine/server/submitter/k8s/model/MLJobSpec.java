package org.apache.submarine.server.submitter.k8s.model;

import java.util.Map;

public interface MLJobSpec {
  Map<MLJobReplicaType, MLJobReplicaSpec> getReplicaSpecs();
  void setReplicaSpecs(Map<MLJobReplicaType, MLJobReplicaSpec> replicaSpecs);
}
