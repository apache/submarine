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

package org.apache.submarine.tony.rpc;

import org.apache.submarine.tony.rpc.impl.TaskStatus;

import java.util.Objects;


/**
 * Contains the name, index, and URL for a task.
 */
public class TaskInfo implements Comparable<TaskInfo> {
  private final String name;   // The name (worker or ps) of the task
  private final String index;  // The index of the task
  private final String url;    // The URL where the logs for the task can be found
  private TaskStatus status = TaskStatus.NEW;

  public TaskInfo(String name, String index, String url) {
    this.name = name;
    this.index = index;
    this.url = url;
  }

  public void setState(TaskStatus status) {
    this.status = status;
  }

  public String getName() {
    return name;
  }

  public String getIndex() {
    return index;
  }

  public String getUrl() {
    return url;
  }

  public TaskStatus getStatus() {
    return status;
  }

  @Override
  public int compareTo(TaskInfo other) {
    if (!this.name.equals(other.name)) {
      return this.name.compareTo(other.name);
    }
    return Integer.valueOf(this.index).compareTo(Integer.valueOf(other.index));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    TaskInfo taskInfo = (TaskInfo) o;
    return Objects.equals(name, taskInfo.name)
            && Objects.equals(index, taskInfo.index)
            && Objects.equals(url, taskInfo.url)
            && Objects.equals(status, taskInfo.getStatus());
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, index, url, status);
  }

  @Override
  public String toString() {
    return String.format(
        "[TaskInfo] name: %s index: %s url: %s status: %s",
        this.name, this.index, this.url, this.status.toString());
  }
}
