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

import org.apache.submarine.server.api.spec.ExperimentSpec;

import io.kubernetes.client.models.V1PodSpec;

public abstract class GitCodeLocalizer extends AbstractCodeLocalizer {

  public void localize(V1PodSpec podSpec) {
    super.localize(podSpec);
  }

  public GitCodeLocalizer(ExperimentSpec experimentSpec) {
    super(experimentSpec);
  }

  public static CodeLocalizer getGitCodeLocalizer(
      ExperimentSpec experimentSpec) {

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
