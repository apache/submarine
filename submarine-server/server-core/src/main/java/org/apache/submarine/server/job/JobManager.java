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

package org.apache.submarine.server.job;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.ws.rs.core.Response.Status;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.SubmitterManager;
import org.apache.submarine.server.api.job.JobSubmitter;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.job.JobId;
import org.apache.submarine.server.api.spec.JobSpec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * It's responsible for managing the jobs CRUD and cache them
 */
public class JobManager {
  private static final Logger LOG = LoggerFactory.getLogger(JobManager.class);

  private static volatile JobManager manager;

  private final AtomicInteger jobCounter = new AtomicInteger(0);

  /**
   * Used to cache the specs by the job id.
   *  key: the string of job id
   *  value: Job object
   */
  private final ConcurrentMap<String, Job> cachedJobMap = new ConcurrentHashMap<>();

  private final JobSubmitter submitter;

  /**
   * Get the singleton instance
   * @return object
   */
  public static JobManager getInstance() {
    if (manager == null) {
      synchronized (JobManager.class) {
        if (manager == null) {
          manager = new JobManager(SubmitterManager.loadSubmitter());
        }
      }
    }
    return manager;
  }

  private JobManager(JobSubmitter submitter) {
    this.submitter = submitter;
  }

  /**
   * Create job
   * @param spec job spec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Job createJob(JobSpec spec) throws SubmarineRuntimeException {
    checkSpec(spec);
    Job job = submitter.createJob(spec);
    job.setJobId(generateJobId());
    job.setSpec(spec);
    cachedJobMap.putIfAbsent(job.getJobId().toString(), job);
    return job;
  }

  private JobId generateJobId() {
    return JobId.newInstance(SubmarineServer.getServerTimeStamp(), jobCounter.incrementAndGet());
  }

  /**
   * Get job
   * @param id job id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Job getJob(String id) throws SubmarineRuntimeException {
    checkJobId(id);
    Job job = cachedJobMap.get(id);
    JobSpec spec = job.getSpec();
    Job patchJob = submitter.findJob(spec);
    job.rebuild(patchJob);
    return job;
  }

  /**
   * List jobs
   * @param status job status, if null will return all jobs
   * @return job list
   * @throws SubmarineRuntimeException the service error
   */
  public List<Job> listJobsByStatus(String status) throws SubmarineRuntimeException {
    List<Job> jobList = new ArrayList<>();
    for (Map.Entry<String, Job> entry : cachedJobMap.entrySet()) {
      Job job = entry.getValue();
      JobSpec spec = job.getSpec();
      Job patchJob = submitter.findJob(spec);
      LOG.info("Found job: {}", patchJob.getStatus());
      if (status == null || status.toLowerCase().equals(patchJob.getStatus().toLowerCase())) {
        job.rebuild(patchJob);
        jobList.add(job);
      }
    }
    LOG.info("List job: {}", jobList.size());
    return jobList;
  }

  /**
   * Patch the job
   * @param id job id
   * @param spec job spec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Job patchJob(String id, JobSpec spec) throws SubmarineRuntimeException {
    checkJobId(id);
    checkSpec(spec);
    Job job = cachedJobMap.get(id);
    Job patchJob = submitter.patchJob(spec);
    job.setSpec(spec);
    job.rebuild(patchJob);
    return job;
  }

  /**
   * Delete job
   * @param id job id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Job deleteJob(String id) throws SubmarineRuntimeException {
    checkJobId(id);
    Job job = cachedJobMap.remove(id);
    JobSpec spec = job.getSpec();
    Job patchJob = submitter.deleteJob(spec);
    job.rebuild(patchJob);
    return job;
  }

  private void checkSpec(JobSpec spec) throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.OK.getStatusCode(), "Invalid job spec.");
    }
  }

  private void checkJobId(String id) throws SubmarineRuntimeException {
    JobId jobId = JobId.fromString(id);
    if (jobId == null || !cachedJobMap.containsKey(id)) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(), "Not found job.");
    }
  }
}
