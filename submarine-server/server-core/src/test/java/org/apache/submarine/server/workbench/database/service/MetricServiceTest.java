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
      metricService.deleteById(metric.getId());
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
    boolean result = metricService.insert(metric);
    assertNotEquals(result, -1);
    List<Metric> metricList = metricService.selectAll();

    assertEquals(metricList.size(), 1);

    Metric metricDb = metricList.get(0);
    compareMetrics(metric, metricDb);

    Metric metricDb2 = metricService.selectById("" + result);
    compareMetrics(metric, metricDb2);
  }

  @Test
  public void testUpdateJob() throws Exception {
    Metric metric = new Metric();
    metric.setMetric_key("score");
    metric.setValue((float) 0.666667);
    metric.setWorker_index("worker-2");
    metric.setTimestamp(new BigInteger("1569139525098"));
    metric.setStep(0);
    metric.setIs_nan(0);
    metric.setJob_name("application_1234");

    boolean result = metricService.insert(metric);
    assertTrue(result);

    metric.setMetric_key("scoreNew");
    metric.setValue((float) 0.766667);
    metric.setWorker_index("worker-New");
    metric.setTimestamp(new BigInteger("2569139525098"));
    metric.setStep(1);
    metric.setIs_nan(1);
    metric.setJob_name("application_1234New");

    boolean editResult = metricService.update(metric);
    assertTrue(editResult);

    Metric metricDb2 = metricService.selectById("" + result);
    compareMetrics(metric, metricDb2);
  }

  @Test
  public void delete() throws Exception {
    Metric metric = new Metric();
    metric.setMetric_key("score");
    metric.setValue((float) 0.666667);
    metric.setWorker_index("worker-2");
    metric.setTimestamp(new BigInteger("1569139525098"));
    metric.setStep(0);
    metric.setIs_nan(0);
    metric.setJob_name("application_1234");

    boolean result = metricService.insert(metric);
    assertTrue(result);

    boolean deleteResult = metricService.deleteById("" + result);
    assertTrue(deleteResult);
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
