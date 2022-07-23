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

package org.apache.submarine.server.api.spec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.submarine.server.k8s.utils.K8sUtils;

/**
 * ExperimentMeta is metadata that all experiments must have.
 */
public class ExperimentMeta {

  public static final String SUBMARINE_EXPERIMENT_NAME = "submarine-experiment-name";

  private String experimentId;
  private String name;
  private final String namespace;
  private String framework;
  private String cmd;
  private Map<String, String> envVars = new HashMap<>();
  private List<String> tags = new ArrayList<>();

  public ExperimentMeta() {
    namespace = K8sUtils.getNamespace();
  }

  /**
   * Get the experiment name which is unique within a namespace.
   * @return experiment name
   */
  public String getName() {
    return name;
  }

  /**
   * Name must be unique within a namespace. Is required when creating experiment.
   * @param name experiment name
   */
  public void setName(String name) {
    this.name = name;
  }

  /**
   * Get the experiment id which is unique within a namespace.
   * @return experiment id
   */
  public String getExperimentId() {
    return experimentId;
  }

  /**
   * experiment id must be unique within a namespace. Is required when creating experiment.
   * @param experimentId experiment id
   */
  public void setExperimentId(String experimentId) {
    this.experimentId = experimentId;
  }

  /**
   * Get the namespace which defines the isolated space for each experiment.
   * @return namespace
   */
  public String getNamespace() {
    return namespace;
  }

  /**
   * Namespace defines the space within each name must be unique.
   * @param namespace namespace
   */
  public void setNamespace(String namespace) {
    // TODO(kevin85421): Remove the function
    return;
  }

  public String getFramework() {
    return framework;
  }

  public void setFramework(String framework) {
    this.framework = framework;
  }

  /**
   * Get the entry command for task.
   * @return cmd
   */
  public String getCmd() {
    return cmd;
  }

  /**
   * The entry command for all tasks if the {@link ExperimentTaskSpec#getCmd()} not specified
   * the cmd replaced with it.
   * @param cmd entry command for task
   */
  public void setCmd(String cmd) {
    this.cmd = cmd;
  }

  /**
   * The default env vars for task. If the @{@link ExperimentTaskSpec#getEnvVars()} not specified
   * replaced with it.
   * @return env vars
   */
  public Map<String, String> getEnvVars() {
    return envVars;
  }

  public void setEnvVars(Map<String, String> envVars) {
    this.envVars = envVars;
  }

  /**
   * The default tag list for task. If the @{@link ExperimentTaskSpec#getEnvVars()} not specified
   * replaced with it.
   * @return tags
   */
  public List<String> getTags() {
    return tags;
  }

  public void setTags(List<String> tags) {
    this.tags = tags;
  }

  /**
   * The {@link ExperimentMeta#framework} should be one of the below supported framework name.
   */
  public enum SupportedMLFramework {
    TENSORFLOW("tensorflow"),
    PYTORCH("pytorch"),
    XGBOOST("xgboost"),
    UNKNOWN("known");

    private final String name;

    SupportedMLFramework(String frName) {
      this.name = frName;
    }

    public String getName() {
      return name;
    }

    public static String[] names() {
      SupportedMLFramework[] frameworks = values();
      String[] names = new String[frameworks.length];
      for (int i = 0; i < frameworks.length; i++) {
        names[i] = frameworks[i].name();
      }
      return names;
    }

    public static SupportedMLFramework valueOfName(String name) {
      for (SupportedMLFramework framework : values()) {
        if (framework.getName().equalsIgnoreCase(name)) {
          return framework;
        }
      }
      return UNKNOWN;
    }
  }

  @Override
  public String toString() {
    return "ExperimentMeta{" +
      "name='" + name + '\'' +
      ", experimentId='" + experimentId + '\'' +
      ", namespace='" + namespace + '\'' +
      ", framework='" + framework + '\'' +
      ", cmd='" + cmd + '\'' +
      ", envVars=" + envVars +
      ", tags=" + tags +
      '}';
  }
}
