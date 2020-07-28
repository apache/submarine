/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.submarine.server.submitter.k8s.experiment.codelocalizer;

import java.util.ArrayList;
import java.util.List;

import org.apache.submarine.server.api.spec.ExperimentSpec;

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
