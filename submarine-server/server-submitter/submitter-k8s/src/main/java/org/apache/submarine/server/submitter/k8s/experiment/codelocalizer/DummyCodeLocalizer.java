package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import org.apache.submarine.server.api.spec.ExperimentSpec;

import io.kubernetes.client.models.V1PodSpec;

public class DummyCodeLocalizer extends AbstractCodeLocalizer {

  public DummyCodeLocalizer(ExperimentSpec experimentSpec) {
    super(experimentSpec);
  }

  @Override
  public void localize(V1PodSpec podSpec) {
    // code specific logic here
  }
}
