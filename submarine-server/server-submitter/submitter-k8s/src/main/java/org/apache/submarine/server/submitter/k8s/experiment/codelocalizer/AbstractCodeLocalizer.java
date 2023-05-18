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

import org.apache.submarine.server.api.exception.InvalidSpecException;

import io.kubernetes.client.openapi.models.V1EmptyDirVolumeSource;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1Volume;
import org.apache.submarine.server.api.spec.CodeSpec;

public abstract class AbstractCodeLocalizer implements CodeLocalizer {

  public static final String CODE_LOCALIZER_PATH = "/code";
  public static final String CODE_LOCALIZER_MOUNT_NAME = "code-dir";
  public static final String CODE_LOCALIZER_INIT_CONTAINER_NAME = "code-localizer";
  public static final String CODE_LOCALIZER_PATH_ENV_VAR = "CODE_PATH";

  @Override
  public void localize(V1PodSpec podSpec) {
    V1Volume volume = new V1Volume();
    volume.setName(CODE_LOCALIZER_MOUNT_NAME);
    volume.setEmptyDir(new V1EmptyDirVolumeSource());
    podSpec.addVolumesItem(volume);
  }

  public static CodeLocalizer getCodeLocalizer(CodeSpec codeSpec)
      throws InvalidSpecException {
    CodeLocalizerModes syncMode = CodeLocalizerModes.valueOfSyncMode(codeSpec.getSyncMode());
    if (syncMode.equals(CodeLocalizerModes.GIT)) {
      return GitCodeLocalizer.getGitCodeLocalizer(codeSpec.getGit());
    } else {
      return new DummyCodeLocalizer();
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

    /**
     * Get CodeLocalizerModes by code key
     */
    public static CodeLocalizerModes valueOfSyncMode(String key) {
      for (CodeLocalizerModes clm : values()) {
        if (clm.mode.equals(key)) {
          return clm;
        }
      }
      return GIT;
    }
  }
}
