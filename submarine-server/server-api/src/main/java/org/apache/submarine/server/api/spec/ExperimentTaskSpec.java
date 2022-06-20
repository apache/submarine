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

import java.util.HashMap;
import java.util.Map;

/**
 * The task spec for an experiment that can include several tasks.
 *   Such as:
 *     The TensorFlow experiment should include Ps/Worker/Chief/Evaluator;
 *     The PyTorch experiment should include Master/Worker.
 */
public class ExperimentTaskSpec {
  private Integer replicas = 1;
  private String resources;

  private String name;
  private String image;
  private String cmd;
  private Map<String, String> envVars;

  // should be ignored in JSON Serialization
  private Map<String, String> resourceMap;

  public ExperimentTaskSpec() {

  }

  /**
   * Get the number of desired tasks.
   * @return number
   */
  public Integer getReplicas() {
    return replicas;
  }

  /**
   * Set the number of desired tasks.
   * @param replicas number
   */
  public void setReplicas(Integer replicas) {
    this.replicas = replicas;
  }

  /**
   * Get the resources. Formatter: cpu=%s,memory=%s,nvidia.com/gpu=%s
   * Resource type list:
   *   <ul>
   *     <li>cpu: In units of core. It known as vcore in YARN.</li>
   *     <li>memory: In units of bytes. Using one of these suffixes: E, P, T, G, M, K</li>
   *     <li>nvidia.com/gpu: </li>
   *   </ul>
   * Such as: cpu=4,memory=2048M,nvidia.com/gpu=1
   * @return resource format string
   */
  public String getResources() {
    return resources;
  }

  /**
   * Set the limit resources for task. Formatter: cpu=%s,memory=%s,nvidia.com/gpu=%s
   * Resource type list:
   *   <ul>
   *     <li>cpu: In units of core. It known as vcore in YARN.</li>
   *     <li>memory: In units of bytes. Using one of these suffixes: E, P, T, G, M, K</li>
   *     <li>nvidia.com/gpu: </li>
   *   </ul>
   * @param resources resource, such as: cpu=4,memory=2048M,nvidia.com/gpu=1
   */
  public void setResources(String resources) {
    this.resources = resources;
    parseResources();
  }

  /**
   * Get the task name.
   * @return task name
   */
  public String getName() {
    return name;
  }

  /**
   * Set the task name
   * @param name task name
   */
  public void setName(String name) {
    this.name = name;
  }

  /**
   * Get the image to start container for running task.
   * @return image
   */
  public String getImage() {
    return image;
  }

  /**
   * Set the task image
   * @param image image
   */
  public void setImage(String image) {
    this.image = image;
  }

  /**
   * Get the entry cmd.
   * @return cmd
   */
  public String getCmd() {
    return cmd;
  }

  /**
   * Set the entry command to start the tasks
   * @param cmd cmd
   */
  public void setCmd(String cmd) {
    this.cmd = cmd;
  }

  /**
   * Get env vars.
   * @return map
   */
  public Map<String, String> getEnvVars() {
    return envVars;
  }

  /**
   * Set the env vars for task
   * @param envVars env map
   */
  public void setEnvVars(Map<String, String> envVars) {
    this.envVars = envVars;
  }

  private void parseResources() {
    if (resources != null) {
      resourceMap = new HashMap<>();
      for (String item : resources.split(",")) {
        String[] r = item.split("=");
        if (r.length == 2) {
          resourceMap.put(r[0], r[1]);
        }
      }
    }
  }

  /**
   * Get cpu resource
   * @return cpu
   */
  public String getCpu() {
    return resourceMap.get("cpu");
  }

  /**
   * Get memory resource
   * @return memory
   */
  public String getMemory() {
    return resourceMap.get("memory");
  }

  /**
   * Get gpu resource
   * @return gpu
   */
  public String getGpu() {
    return resourceMap.get("nvidia.com/gpu");
  }

}
