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

import java.util.List;

import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1EnvVar;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1SecurityContext;
import io.kubernetes.client.openapi.models.V1VolumeMount;

public class SSHGitCodeLocalizer extends GitCodeLocalizer {

  public static final String GIT_SECRET_NAME = "git-creds";
  public static final int GIT_SECRET_MODE = 0400;
  public static final String GIT_SECRET_MOUNT_NAME = "git-secret";
  public static final String GIT_SECRET_PATH = "/etc/git-secret";
  public static final long GIT_SYNC_USER = 65533L;
  public static final String GIT_SYNC_SSH_NAME = "GIT_SYNC_SSH";
  public static final String GIT_SYNC_SSH_VALUE = "true";

  public SSHGitCodeLocalizer(String url) {
    super(url);
  }

  @Override
  public void localize(V1PodSpec podSpec) {
    super.localize(podSpec);
    for (V1Container container : podSpec.getInitContainers()) {
      if (container.getName().equals(CODE_LOCALIZER_INIT_CONTAINER_NAME)) {
        List<V1EnvVar> gitSyncEnvVars = container.getEnv();
        V1EnvVar sshEnv = new V1EnvVar();
        sshEnv.setName(GIT_SYNC_SSH_NAME);
        sshEnv.setValue(GIT_SYNC_SSH_VALUE);
        gitSyncEnvVars.add(sshEnv);

        List<V1VolumeMount> mounts = container.getVolumeMounts();
        V1VolumeMount mount = new V1VolumeMount();
        mount.setName(GIT_SECRET_MOUNT_NAME);
        mount.setMountPath(GIT_SECRET_PATH);
        mount.setReadOnly(true);
        mounts.add(mount);

        V1SecurityContext containerSecurityContext =
            new V1SecurityContext();
        containerSecurityContext
            .setRunAsUser(SSHGitCodeLocalizer.GIT_SYNC_USER);
        container.setSecurityContext(containerSecurityContext);
      }
    }
  }
}
