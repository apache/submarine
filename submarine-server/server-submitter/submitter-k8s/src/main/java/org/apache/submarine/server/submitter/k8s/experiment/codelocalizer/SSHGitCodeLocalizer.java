package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;

import io.kubernetes.client.models.V1Container;
import io.kubernetes.client.models.V1PodSpec;

public class SSHGitCodeLocalizer extends GitCodeLocalizer {

  public SSHGitCodeLocalizer(ExperimentSpec experimentSpec) {
    super(experimentSpec);
  }

  @Override
  public void localize(V1PodSpec podSpec) {
  }

}
