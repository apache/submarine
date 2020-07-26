package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import io.kubernetes.client.models.V1PodSpec;

public interface CodeLocalizer {

  /**
   * Create K8's Init container to sync the code
   */
  void localize(V1PodSpec podSpec);

}
