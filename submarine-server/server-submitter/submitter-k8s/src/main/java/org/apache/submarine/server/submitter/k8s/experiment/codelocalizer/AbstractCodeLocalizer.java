package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import org.apache.submarine.server.api.spec.ExperimentSpec;

import io.kubernetes.client.models.V1EmptyDirVolumeSource;
import io.kubernetes.client.models.V1PodSpec;
import io.kubernetes.client.models.V1Volume;

public abstract class AbstractCodeLocalizer implements CodeLocalizer {

  protected ExperimentSpec experimentSpec;
  
  public AbstractCodeLocalizer(ExperimentSpec experimentSpec) {
    this.experimentSpec = experimentSpec;
  }
  
  @Override
  public void localize(V1PodSpec podSpec) {
    
    V1Volume volume = new V1Volume();
    volume.setName("code-dir");
    volume.setEmptyDir(new V1EmptyDirVolumeSource());
    podSpec.addVolumesItem(volume);
    
  }

  public static CodeLocalizer getCodeLocalizer(ExperimentSpec experimentSpec) {

    String syncMode = experimentSpec.getCode().getSyncMode();
    if (syncMode.equals(CodeLocalizerModes.GIT.getMode())) {
      return GitCodeLocalizer.getGitCodeLocalizer(experimentSpec);
    } else {
      return new DummyCodeLocalizer(experimentSpec);
    }
  }

  public enum CodeLocalizerModes {

    GIT("git"), HDFS("hdfs"), NFS("nfs"), S3("s3");

    private final String mode;

    CodeLocalizerModes(String mode) {
      this.mode = mode;
    }

    public String getMode() {
      return this.mode;
    }
  }
}
