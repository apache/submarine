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
package org.apache.submarine.server.database.workbench.service;

import org.apache.submarine.server.database.workbench.entity.JobEntity;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class JobServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(JobServiceTest.class);
  JobService jobService = new JobService();

  @After
  public void removeAllRecord() throws Exception {
    List<JobEntity> jobList = jobService.queryJobList(null, "create_time", "desc", 0, 100);
    LOG.info("jobList.size():{}", jobList.size());
    for (JobEntity job : jobList) {
      jobService.delete(job.getId());
    }
  }

  @Test
  public void testSelectJob() throws Exception {
    JobEntity job = new JobEntity();
    job.setJobId("9e93caeb-8a08-4278-a10a-ff60d5835716");
    job.setJobName("mnist");
    job.setJobNamespace("submarine");
    job.setUserName("JobServiceTest-UserName");
    job.setJobType("TFJob");
    job.setJobStatus("Finished");
    job.setJobFinalStatus("Succeeded");
    job.setCreateBy("JobServiceTest-UserName");

    Boolean ret = jobService.add(job);
    assertTrue(ret);

    List<JobEntity> jobList = jobService.queryJobList(
            "JobServiceTest-UserName",
            "create_time",
            "desc",
            0,
            100);
    assertEquals(jobList.size(), 1);

    JobEntity jobDb = jobList.get(0);
    compareJobs(job, jobDb);

    JobEntity jobDb2 = jobService.selectByJobId("9e93caeb-8a08-4278-a10a-ff60d5835716");
    compareJobs(job, jobDb2);
  }

  @Test
  public void testUpdateJob() throws Exception {
    JobEntity job = new JobEntity();
    job.setJobId("9e93caeb-8a08-4278-a10a-ff60d5835716");
    job.setJobName("mnist");
    job.setJobNamespace("submarine");
    job.setUserName("JobServiceTest-UserName");
    job.setJobType("TFJob");
    job.setJobStatus("Finished");
    job.setJobFinalStatus("Succeeded");
    job.setCreateBy("JobServiceTest-UserName");

    Boolean ret = jobService.add(job);
    assertTrue(ret);

    job.setJobName("mnistNew");
    job.setJobNamespace("submarineNew");
    job.setUserName("JobServiceTest-UserNameNew");
    job.setJobType("TFJobNew");
    job.setJobStatus("Running");
    job.setJobFinalStatus("");
    job.setUpdateBy("JobServiceTest-UserNameNew");

    boolean editRet = jobService.updateByPrimaryKeySelective(job);
    assertTrue(editRet);

    JobEntity jobDb2 = jobService.selectByJobId("9e93caeb-8a08-4278-a10a-ff60d5835716");
    compareJobs(job, jobDb2);
  }

  @Test
  public void delete() throws Exception {
    JobEntity job = new JobEntity();
    job.setJobId("9e93caeb-8a08-4278-a10a-ff60d5835716");
    job.setJobName("mnist");
    job.setJobNamespace("submarine");
    job.setUserName("JobServiceTest-UserName");
    job.setJobType("TFJob");
    job.setJobStatus("Finished");
    job.setJobFinalStatus("Succeeded");
    job.setCreateBy("JobServiceTest-UserName");

    Boolean ret = jobService.add(job);
    assertTrue(ret);

    Boolean deleteRet = jobService.delete(job.getId());
    assertTrue(deleteRet);
  }

  private void compareJobs(JobEntity job, JobEntity jobDb) {
    assertEquals(job.getJobId(), jobDb.getJobId());
    assertEquals(job.getJobName(), jobDb.getJobName());
    assertEquals(job.getJobNamespace(), jobDb.getJobNamespace());
    assertEquals(job.getUserName(), jobDb.getUserName());
    assertEquals(job.getJobType(), jobDb.getJobType());
    assertEquals(job.getJobStatus(), jobDb.getJobStatus());
    assertEquals(job.getJobFinalStatus(), jobDb.getJobFinalStatus());
    assertEquals(job.getCreateBy(), jobDb.getCreateBy());
    assertEquals(job.getUpdateBy(), jobDb.getUpdateBy());
  }
}
