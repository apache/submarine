/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.apache.submarine.tony;

import java.util.List;
import java.util.Map;

/**
 * TFConfig POJO for serialization.
 */
public class TFConfig {

  private Map<String, List<String>> clusterSpec;
  private Task task;

  public static class Task {
    private String type;
    private int index;

    // Jackson needs a default constructor
    Task() { }

    Task(String type, int index) {
      this.type = type;
      this.index = index;
    }

    // Getters required for serialization
    public String getType() {
      return this.type;
    }

    public int getIndex() {
      return this.index;
    }

    // Setters required for deserialization
    public void setType(String type) {
      this.type = type;
    }

    public void setIndex(int index) {
      this.index = index;
    }
  }

  // Jackson needs a default constructor
  TFConfig() { }

  public TFConfig(Map<String, List<String>> clusterSpec, String jobName, int taskIndex) {
    this.clusterSpec = clusterSpec;
    this.task = new Task(jobName, taskIndex);
  }

  // Getters required for serialization
  public Map<String, List<String>> getCluster() {
    return this.clusterSpec;
  }

  public Task getTask() {
    return this.task;
  }

  // Setters required for deserialization
  public void setCluster(Map<String, List<String>> clusterSpec) {
    this.clusterSpec = clusterSpec;
  }

  public void setTask(Task task) {
    this.task = task;
  }
}
