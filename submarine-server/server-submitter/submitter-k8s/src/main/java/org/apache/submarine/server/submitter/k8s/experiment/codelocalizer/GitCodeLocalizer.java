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

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1VolumeMount;
import org.apache.submarine.server.api.exception.InvalidSpecException;

import org.apache.submarine.server.api.spec.code.GitCodeSpec;
import org.apache.submarine.server.submitter.k8s.util.K8sResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class GitCodeLocalizer extends AbstractCodeLocalizer {

  private static final Logger LOG = LoggerFactory.getLogger(GitCodeLocalizer.class);

  public static final String GIT_SYNC_IMAGE = "apache/submarine:git-sync-3.1.6";

  private final GitCodeSpec codeSpec;

  public GitCodeLocalizer(GitCodeSpec codeSpec) {
    this.codeSpec = codeSpec;
  }

  public GitCodeSpec getCodeSpec() {
    return codeSpec;
  }

  public void localize(V1PodSpec podSpec) {

    V1Container container = new V1Container();
    container.setName(CODE_LOCALIZER_INIT_CONTAINER_NAME);
    container.setImage(GIT_SYNC_IMAGE);

    // Add some default git sync envs
    // The current git environment variables supported by git-syn can be referred to:
    // https://github.com/kubernetes/git-sync/blob/v3.1.6/cmd/git-sync/main.go
    List<V1EnvVar> gitSyncEnvVars = new ArrayList<V1EnvVar>();

    gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_REPO", getCodeSpec().getUrl()));
    gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_ROOT", CODE_LOCALIZER_PATH));
    // Our scenario is usually to download the latest code once and execute it,
    // so we set depth to 1 to prevent the code base from getting too large
    gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_DEPTH", "1"));
    gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_DEST", "current"));
    // Download first and then exit
    gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_ONE_TIME", "true"));
    // branch
    gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_BRANCH", getCodeSpec().getBranch()));

    // Add some optional git sync envs
    //  username
    if (getCodeSpec().getUsername() != null) {
      gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_USERNAME", getCodeSpec().getUsername()));
    }
    //  password
    if (getCodeSpec().getPassword() != null) {
      gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SYNC_PASSWORD", getCodeSpec().getPassword()));
    }
    //  accept a self-signed certificate host
    if (getCodeSpec().getTrustCerts()) {
      gitSyncEnvVars.add(K8sResourceUtils.createEnvVar("GIT_SSL_NO_VERIFY", "true"));
    }

    container.setEnv(gitSyncEnvVars);

    V1VolumeMount mount = new V1VolumeMount();
    mount.setName(CODE_LOCALIZER_MOUNT_NAME);
    mount.setMountPath(CODE_LOCALIZER_PATH);

    List<V1VolumeMount> volumeMounts = new ArrayList<V1VolumeMount>();
    volumeMounts.add(mount);

    container.setVolumeMounts(volumeMounts);

    podSpec.addInitContainersItem(container);

    super.localize(podSpec);
  }

  /**
   * Currently, we mainly support https and ssh.
   * By default, we use https.
   */
  public static CodeLocalizer getGitCodeLocalizer(GitCodeSpec gitCodeSpec)
      throws InvalidSpecException {

    String url = gitCodeSpec.getUrl();
    try {
      URI uriParser = new URI(url);
      String scheme = uriParser.getScheme();
      if (scheme.equals(GitCodeLocalizerModes.SSH.getMode())) {
        return new SSHGitCodeLocalizer(gitCodeSpec);
      } else if (scheme.equals(GitCodeLocalizerModes.HTTPS.getMode())) {
        return new HTTPGitCodeLocalizer(gitCodeSpec);
      } else {
        LOG.debug("Unknown url scheme, use https as default localizer.");
        return new HTTPGitCodeLocalizer(gitCodeSpec);
      }
    } catch (URISyntaxException e) {
      throw new InvalidSpecException(
          "Invalid Code Spec: URL is malformed. " + url);
    }
  }

  public enum GitCodeLocalizerModes {

    HTTPS("https"), SSH("ssh");

    private final String mode;

    GitCodeLocalizerModes(String mode) {
      this.mode = mode;
    }

    public String getMode() {
      return this.mode;
    }
  };

}
