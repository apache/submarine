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

package org.apache.submarine.server.submitter.yarnservice;

import org.apache.hadoop.yarn.client.api.AppAdminClient;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.service.api.records.Service;
import org.apache.hadoop.yarn.service.utils.ServiceApiUtil;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.api.JobStatus;
import org.apache.submarine.commons.runtime.JobMonitor;
import org.apache.submarine.server.submitter.yarnservice.builder.JobStatusBuilder;

import java.io.IOException;

public class YarnServiceJobMonitor extends JobMonitor {
  private volatile AppAdminClient serviceClient = null;

  public YarnServiceJobMonitor(ClientContext clientContext) {
    super(clientContext);
  }

  @Override
  public JobStatus getTrainingJobStatus(String jobName)
      throws IOException, YarnException {
    if (this.serviceClient == null) {
      synchronized (this) {
        if (this.serviceClient == null) {
          this.serviceClient = YarnServiceUtils.createServiceClient(
              clientContext.getYarnConfig());
        }
      }
    }
    String appStatus = serviceClient.getStatusString(jobName);
    Service serviceSpec = ServiceApiUtil.jsonSerDeser.fromJson(appStatus);
    JobStatus jobStatus = JobStatusBuilder.fromServiceSpec(serviceSpec);
    return jobStatus;
  }

  @Override
  public void cleanup() throws IOException {
    if (this.serviceClient != null) {
      this.serviceClient.close();
    }
  }
}
