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
package org.apache.submarine.server.workbench.database.service;

import org.apache.submarine.server.workbench.database.entity.Metric;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigInteger;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class MetricServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(MetricServiceTest.class);
  MetricService metricService = new MetricService();

  @After
  public void removeAllRecord() throws Exception {
    List<Metric> metricList = metricService.selectAll();
    LOG.info("jobList.size():{}", metricList.size());
    for (Metric metric : metricList) {
      metricService.deleteByPrimaryKey(metric.getId());
    }
  }

/*
# +-------+----------+--------------+---------------+------+--------+------------------+
# | key   | value    | worker_index | timestamp     | step | is_nan | job_name         |
# +-------+----------+--------------+---------------+------+--------+------------------+
# | score | 0.666667 | worker-1     | 1569139525097 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569149139731 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569169376482 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569236290721 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569236466722 |    0 |      0 | application_1234 |
# +-------+----------+--------------+---------------+------+--------+------------------+
*/

  @Test
  public void testSelectMetric() throws Exception {
    Metric metric = new Metric();
    metric.setMetric_key("score");
    metric.setValue((float) 0.666667);
    metric.setWorker_index("worker-1");
    metric.setTimestamp(new BigInteger("1569139525097"));
    metric.setStep(0);
    metric.setIs_nan(0);
    metric.setJob_name("application_1234");
    int result = metricService.insert(metric);
    assertNotEquals(result, -1);
    List<Metric> metricList = metricService.selectAll();

    assertEquals(metricList.size(), 1);

    Metric metricDb = metricList.get(0);
    compareMetrics(metric, metricDb);

    Metric metricDb2 = metricService.selectByPrimaryKey("" + result);
    compareMetrics(metric, metricDb2);
  }

  @Test
  public void testUpdateJob() throws Exception {
    Job job = new Job();
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

    Job jobDb2 = jobService.selectByJobId("9e93caeb-8a08-4278-a10a-ff60d5835716");
    compareJobs(job, jobDb2);
  }

  @Test
  public void delete() throws Exception {
    Job job = new Job();
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

  private void compareJobs(Job job, Job jobDb) {
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

  private void compareMetrics(Metric metric, Metric metricDb) {
    assertEquals(metric.getId(), metricDb.getId());
    assertEquals(metric.getIs_nan(), metricDb.getIs_nan());
    assertEquals(metric.getJob_name(), metricDb.getJob_name());
    assertEquals(metric.getMetric_key(), metricDb.getMetric_key());
    assertEquals(metric.getStep(), metricDb.getStep());
    assertEquals(metric.getTimestamp(), metricDb.getTimestamp());
    assertEquals(metric.getValue(), metricDb.getValue());
    assertEquals(metric.getWorker_index(), metricDb.getWorker_index());
  }
}
