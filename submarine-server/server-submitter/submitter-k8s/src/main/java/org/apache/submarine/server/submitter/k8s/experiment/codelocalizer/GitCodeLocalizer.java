package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;

import io.kubernetes.client.models.V1PodSpec;

public abstract class GitCodeLocalizer extends AbstractCodeLocalizer {

  public void localize(V1PodSpec podSpec) {
    super.localize(podSpec);
  }

  public GitCodeLocalizer(ExperimentSpec experimentSpec) {
    super(experimentSpec);
  }

  public static CodeLocalizer getGitCodeLocalizer(ExperimentSpec experimentSpec) {
    
    String url = experimentSpec.getCode().getUrl();
    if (url.contains(GitCodeLocalizerModes.HTTP.getMode())) {
      return new HTTPGitCodeLocalizer(experimentSpec);
    } else if (url.contains(GitCodeLocalizerModes.SSH.getMode())) {
      return new SSHGitCodeLocalizer(experimentSpec);
    } else {
      return new DummyCodeLocalizer(experimentSpec);
    }
  }

  public enum GitCodeLocalizerModes {

    HTTP("https"), SSH("ssh");

    private final String mode;

    GitCodeLocalizerModes(String mode) {
      this.mode = mode;
    }

    public String getMode() {
      return this.mode;
    }
  };

}
