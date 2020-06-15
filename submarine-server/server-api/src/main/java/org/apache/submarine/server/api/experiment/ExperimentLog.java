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

package org.apache.submarine.server.api.experiment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ExperimentLog {
  private String experimentId;
  private List<PodLog> logContent;

  class PodLog {

    String podName;
    List<String> podLog = new ArrayList<String>();

    PodLog(String name, String log) {
      this.podName = name;
      this.podLog = new ArrayList<String>();
      addLog(log);
    }
    void addLog(String log) {
      if ( log != null)
        this.podLog.addAll(Arrays.asList(log.split("\n")));
    }
  }

  public ExperimentLog() {
    logContent = new ArrayList<PodLog>();
  }

  public void setExperimentId(String experimentId) {
    this.experimentId = experimentId;
  }

  public String getExperimentId() {
    return experimentId;
  }

  public void addPodLog(String name, String log) {
    for (PodLog podlog : logContent) {
      if (podlog.podName.equals(name))
      {
        podlog.addLog(log);
        return;
      }
    }
    logContent.add(new PodLog(name, log));
  }

  public void clearPodLog() {
    logContent.clear();
  }
}
