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

package org.apache.submarine.server.submitter.yarn;

import com.linkedin.tony.TonyClient;
import com.linkedin.tony.client.TaskUpdateListener;
import com.linkedin.tony.rpc.TaskInfo;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.api.JobStatus;
import org.apache.submarine.commons.runtime.JobMonitor;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * An implementation of JobMonitor with TonY library.
 */
public class YarnJobMonitor extends JobMonitor implements TaskUpdateListener {
  private Set<TaskInfo> taskInfos = new HashSet<>();

  public YarnJobMonitor(ClientContext clientContext, TonyClient client) {
    super(clientContext);
    client.addListener(this);
  }

  @Override
  public JobStatus getTrainingJobStatus(String jobName)
      throws IOException, YarnException {
    JobStatus jobStatus = JobStatusBuilder.fromTaskInfoSet(taskInfos);
    jobStatus.setJobName(jobName);
    return jobStatus;
  }

  @Override
  public void onTaskInfosUpdated(Set<TaskInfo> taskInfoSet) {
    this.taskInfos = taskInfoSet;
  }
}
