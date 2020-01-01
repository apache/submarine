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
 * The machine learning related spec for job.
 */
public class JobLibrarySpec {
  /**
   * Machine Learning Framework name. Such as: TensorFlow/PyTorch etc.
   */
  private String name;

  /**
   * The version of ML framework. Such as: 2.1.0
   */
  private String version;

  /**
   * The public image used for each task if not specified. Such as: apache/submarine
   */
  private String image;

  /**
   * The public entry cmd for the task if not specified.
   */
  private String cmd;

  /**
   * The public env vars for the task if not specified.
   */
  private Map<String, String> envVars;

  public JobLibrarySpec() {

  }

  /**
   * Get the name of the machine learning library. Such as: TensorFlow or PyTorch
   * @return the library's name
   */
  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  /**
   * Get the version of the library.
   * @return the library's version
   */
  public String getVersion() {
    return version;
  }

  public void setVersion(String version) {
    this.version = version;
  }

  /**
   * Get the image for the machine learning job. If the {@link JobTaskSpec#getImage()} not
   * specified the image replaced with it.
   * @return image url link.
   */
  public String getImage() {
    return image;
  }

  public void setImage(String image) {
    this.image = image;
  }

  /**
   * The default entry command for job task. If the {@link JobTaskSpec#getCmd()} not specified
   * the cmd replaced with it.
   * @return cmd
   */
  public String getCmd() {
    return cmd;
  }

  public void setCmd(String cmd) {
    this.cmd = cmd;
  }

  /**
   * The default env vars for job task. If the @{@link JobTaskSpec#getEnvVars()} not specified
   * replaced with it.
   * @return env vars
   */
  public Map<String, String> getEnvVars() {
    return envVars;
  }

  public void setEnvVars(Map<String, String> envVars) {
    this.envVars = envVars;
  }

  public boolean validate() {
    return name != null && image != null && cmd != null;
  }
}
