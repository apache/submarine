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

package org.apache.submarine.server;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.JobSubmitter;
import org.apache.submarine.server.api.exception.UnsupportedJobTypeException;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.job.JobId;
import org.apache.submarine.server.api.job.spec.MLJobSpec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class JobManager {
  private static final Logger LOG = LoggerFactory.getLogger(JobManager.class);
  private static volatile JobManager manager;

  private final AtomicInteger jobCounter = new AtomicInteger(0);

  private final ConcurrentMap<JobId, Job> jobs = new ConcurrentHashMap<>();

  private SubmitterManager submitterManager;
  private ExecutorService executorService;

  public static JobManager getInstance() {
    if (manager == null) {
      synchronized (JobManager.class) {
        if (manager == null) {
          SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
          SubmitterManager submitterManager = new SubmitterManager(conf);
          manager = new JobManager(submitterManager);
        }
      }
    }
    return manager;
  }

  public JobManager(SubmitterManager submitterManager) {
    this.submitterManager = submitterManager;
    this.executorService = Executors.newFixedThreadPool(50);
  }

  /**
   * Used by REST/RPC service to submit the ML job and obtain a new {@link JobId} for submitting
   * new job.
   *
   * @param spec job spec
   * @return JobId instance
   */
  public JobId submitJob(MLJobSpec spec) {
    JobId jobId = generateJobId();
    executorService.submit(() -> {
      try {
        JobSubmitter submitter = submitterManager.getSubmitterByType("");
        Job job = submitter.submitJob(spec);
        jobs.putIfAbsent(jobId, job);
      } catch (UnsupportedJobTypeException e) {
        LOG.warn(e.getMessage(), e);
      }
    });
    return jobId;
  }

  private JobId generateJobId() {
    return JobId.newInstance(SubmarineServer.getServerTimeStamp(), jobCounter.incrementAndGet());
  }

  public Job getJob(JobId jobId) {
    return jobs.get(jobId);
  }
}
