package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import java.util.ArrayList;
import java.util.List;

import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;

import io.kubernetes.client.models.V1Container;
import io.kubernetes.client.models.V1PodSpec;
import io.kubernetes.client.models.V1VolumeMount;

public class HTTPGitCodeLocalizer extends GitCodeLocalizer {

  public HTTPGitCodeLocalizer(ExperimentSpec experimentSpec) {
    super(experimentSpec);
  }
  
  @Override
  public void localize(V1PodSpec podSpec) {
    
    V1Container container = new V1Container();
    
    container.setName("git-localizer");
    container.setImage("git-sync");
    
    V1VolumeMount mount = new V1VolumeMount();
    mount.setName("code-dir");
    mount.setMountPath("/code");
    
    List<V1VolumeMount> volumeMounts = new ArrayList<V1VolumeMount>();
    volumeMounts.add(mount);
    
    container.setVolumeMounts(volumeMounts);
    
    podSpec.addInitContainersItem(container);
    
    super.localize(podSpec);
  }

}
