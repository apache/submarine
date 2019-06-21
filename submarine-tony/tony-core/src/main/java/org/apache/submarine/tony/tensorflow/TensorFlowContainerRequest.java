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

package org.apache.submarine.tony.tensorflow;


public class TensorFlowContainerRequest {
  private int numInstances;
  private long memory;
  private int vCores;
  private int priority;
  private int gpu;
  private String jobName;

  public TensorFlowContainerRequest(String jobName, int numInstances,
                                    long memory, int vCores, int gpu, int priority) {
    this.numInstances = numInstances;
    this.memory = memory;
    this.vCores = vCores;
    this.priority = priority;
    this.gpu = gpu;
    this.jobName = jobName;
  }

  public TensorFlowContainerRequest(TensorFlowContainerRequest that) {
    this.memory = that.memory;
    this.vCores = that.vCores;
    this.gpu = that.gpu;
    this.priority = that.priority;
    this.jobName = that.jobName;
  }

  public int getNumInstances() {
    return numInstances;
  }

  public long getMemory() {
    return memory;
  }

  public int getVCores() {
    return vCores;
  }

  public int getGPU() {
    return gpu;
  }

  public int getPriority() {
    return priority;
  }

  public String getJobName() {
    return jobName;
  }
}
