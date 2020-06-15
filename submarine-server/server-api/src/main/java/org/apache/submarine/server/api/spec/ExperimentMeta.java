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

import java.util.Map;

/**
 * ExperimentMeta is metadata that all experiments must have.
 */
public class ExperimentMeta {
  private String name;
  private String namespace;
  private String framework;
  private String cmd;
  private Map<String, String> envVars;

  public ExperimentMeta() {

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
    this.namespace = namespace;
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
   * The {@link ExperimentMeta#framework} should be one of the below supported framework name.
   */
  public enum SupportedMLFramework {
    TENSORFLOW("tensorflow"),
    PYTORCH("pytorch");

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
  }
}
