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

public class NotebookPodSpec {

  private Map<String, String> envVars;
  private String resources;

  // should ignored in JSON Serialization
  private transient Map<String, String> resourceMap;

  public NotebookPodSpec() {

  }

  /**
   * Get envVars
   * @return envVars
   */
  public Map<String, String> getEnvVars() {
    return envVars;
  }

  /**
   * Set envVars
   * @param envVars environment variables for container
   */
  public void setEnvVars(Map<String, String> envVars) {
    this.envVars = envVars;
  }

  /**
   * Get the resources for container.
   * Resource type
   * cpu: vCPU/Core
   * memory: E/Ei, P/Pi, T/Ti, G/Gi, M/Mi, K/Ki
   * nvidia.com/gpu: GPUs (not possible to request a fraction of a GPU)
   * such as: cpu=1,memory=2Gi,nvidia.com/gpu=1
   * @return resources resources for container
   */
  public String getResources() {
    return resources;
  }

  /**
   * Set the resource for container
   * Resource type
   * cpu: vCPU/Core
   * memory: E/Ei, P/Pi, T/Ti, G/Gi, M/Mi, K/Ki
   * nvidia.com/gpu: GPUs (not possible to request a fraction of a GPU)
   * such as: cpu=1,memory=2Gi,nvidia.com/gpu=1
   * @param resources resources for container
   */
  public void setResources(String resources) {
    this.resources = resources;
    parseResources();
  }

  /**
   * Parse resources
   */
  public void parseResources() {
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
   * Get the cpu reserved by the notebook server
   * @return String or null
   */
  public String getCpu() {
    return resourceMap.get("cpu");
  }

  /**
   * Get the memory reserved by the notebook server
   * @return String or null
   */
  public String getMemory() {
    return resourceMap.get("memory");
  }

  /**
   * Get the gpu reserved by the notebook server
   * @return String or null
   */
  public String getGpu() {
    return resourceMap.get("nvidia.com/gpu");
  }

}
