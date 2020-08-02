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

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.apache.submarine.server.api.exception.InvalidSpecException;

import io.kubernetes.client.models.V1Container;
import io.kubernetes.client.models.V1EnvVar;
import io.kubernetes.client.models.V1PodSpec;
import io.kubernetes.client.models.V1VolumeMount;

public abstract class GitCodeLocalizer extends AbstractCodeLocalizer {

  public static final String GIT_SYNC_IMAGE = "k8s.gcr.io/git-sync:v3.1.6";
  public static final String GIT_SYNC_ROOT = "/code";
  
  public GitCodeLocalizer(String url) {
    super(url);
  }

  public void localize(V1PodSpec podSpec) {
    
    V1Container container = new V1Container();
    container.setName("git-localizer");
    container.setImage(GIT_SYNC_IMAGE);
    
    V1EnvVar repoEnv = new V1EnvVar();
    repoEnv.setName("GIT_SYNC_REPO");
    repoEnv.setValue(this.getUrl());
    
    V1EnvVar rootEnv = new V1EnvVar();
    rootEnv.setName("GIT_SYNC_ROOT");
    rootEnv.setValue(GIT_SYNC_ROOT);
    
    V1EnvVar destEnv = new V1EnvVar();
    destEnv.setName("GIT_SYNC_DEST");
    destEnv.setValue("current");
    
    V1EnvVar oneTimeEnv = new V1EnvVar();
    oneTimeEnv.setName("GIT_SYNC_ONE_TIME");
    oneTimeEnv.setValue("true");
    
    List<V1EnvVar> gitSyncEnvVars = new ArrayList<V1EnvVar>();
    gitSyncEnvVars.add(repoEnv);
    gitSyncEnvVars.add(rootEnv);
    gitSyncEnvVars.add(destEnv);
    gitSyncEnvVars.add(oneTimeEnv);
    container.setEnv(gitSyncEnvVars);
    
    V1VolumeMount mount = new V1VolumeMount();
    mount.setName("code-dir");
    mount.setMountPath("/code");
    
    List<V1VolumeMount> volumeMounts = new ArrayList<V1VolumeMount>();
    volumeMounts.add(mount);
    
    container.setVolumeMounts(volumeMounts);
    
    podSpec.addInitContainersItem(container);
    
    super.localize(podSpec);
  }
  
  public static CodeLocalizer getGitCodeLocalizer(String url)
      throws InvalidSpecException {

    try {
      URL urlParser = new URL(url);
      String protocol = urlParser.getProtocol();
      if (protocol.equals(GitCodeLocalizerModes.HTTP.getMode())) {
        return new HTTPGitCodeLocalizer(url);
      } else if (protocol.equals(GitCodeLocalizerModes.SSH.getMode())) {
        return new SSHGitCodeLocalizer(url);
      } else {
        return new DummyCodeLocalizer(url);
      }
    } catch (MalformedURLException e) {
      throw new InvalidSpecException(
          "Invalid Code Spec: URL is malformed. " + url);
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
