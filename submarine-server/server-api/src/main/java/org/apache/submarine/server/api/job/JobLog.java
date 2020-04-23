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

package org.apache.submarine.server.api.job;

import java.util.ArrayList;
import java.util.List;

public class JobLog {
  private String jobId;
  private List<podLog> logContent;

  class podLog {
    String podName;
    String podLog;
    podLog(String podName, String podLog) {
      this.podName = podName;
      this.podLog = podLog;
    }
  }

  public JobLog() {
    logContent = new ArrayList<podLog>();
  }
  
  public void setJobId(String jobId) {
    this.jobId = jobId;
  }
  
  public String getJobId() {
    return jobId;
  }

  public void addPodLog(String name, String log) {
    logContent.add(new podLog(name, log));
  }

  public void clearPodLog() {
    logContent.clear();
  }
}
