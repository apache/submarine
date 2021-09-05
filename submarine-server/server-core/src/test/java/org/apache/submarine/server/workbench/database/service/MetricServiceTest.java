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

import org.apache.submarine.server.experiment.database.ExperimentEntity;
import org.apache.submarine.server.experiment.database.ExperimentService;
import org.apache.submarine.server.workbench.database.entity.Metric;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class MetricServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(MetricServiceTest.class);
  MetricService metricService = new MetricService();
  ExperimentService experimentService = new ExperimentService();

  // Id of metric is a foreign key for experiment id so experiment must be created before test.
  @Before
  public void createExperiment() throws Exception {
    ExperimentEntity entity = new ExperimentEntity();
    String id = "test_application_1234";
    String spec = "{\"value\": 1}";

    entity.setId(id);
    entity.setExperimentSpec(spec);

    experimentService.insert(entity);
  }

  @After
  public void removeAllRecord() throws Exception {
    List<Metric> metricList = metricService.selectAll();
    LOG.info("jobList.size():{}", metricList.size());
    for (Metric metric : metricList) {
      metricService.deleteById(metric.getId());
    }

    experimentService.selectAll().forEach(e -> experimentService.delete(e.getId()));
  }

  @Test
  public void testSelect() throws Exception {
    String dateStr = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    Timestamp timestamp = Timestamp.valueOf(dateStr);

    Metric metric = new Metric();
    metric.setId("test_application_1234");
    metric.setKey("test_score");
    metric.setValue((float) 0.666667);
    metric.setWorkerIndex("test_worker-1");
    metric.setTimestamp(timestamp);
    metric.setStep(0);
    metric.setIsNan(false);
    boolean result = metricService.insert(metric);
    assertTrue(result);
    List<Metric> metricList = metricService.selectAll();

    assertEquals(metricList.size(), 1);

    Metric metricDb = metricList.get(0);
    compareMetrics(metric, metricDb);

    Metric metricDb2 = metricService.selectByPrimaryKeySelective(metric).get(0);
    compareMetrics(metric, metricDb2);
  }

  @Test
  public void testUpdate() throws Exception {
    String dateStr = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    Timestamp timestamp = Timestamp.valueOf(dateStr);

    Metric metric = new Metric();
    metric.setId("test_application_1234");
    metric.setKey("test_score");
    metric.setValue((float) 0.666667);
    metric.setWorkerIndex("test_worker-2");
    metric.setTimestamp(timestamp);
    metric.setStep(0);
    metric.setIsNan(false);
    boolean result = metricService.insert(metric);
    assertTrue(result);

    dateStr = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    Timestamp nextTimestamp = Timestamp.valueOf(dateStr);

    metric.setId("test_application_1234");
    metric.setKey("test_scoreNew");
    metric.setValue((float) 0.766667);
    metric.setWorkerIndex("test_worker-New");
    metric.setTimestamp(nextTimestamp);
    metric.setStep(1);
    metric.setIsNan(true);

    boolean editResult = metricService.update(metric);
    assertTrue(editResult);

    Metric metricDb2 = metricService.selectByPrimaryKeySelective(metric).get(0);
    compareMetrics(metric, metricDb2);
  }

  @Test
  public void testDelete() throws Exception {
    String dateStr = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    Timestamp timestamp = Timestamp.valueOf(dateStr);

    Metric metric = new Metric();
    metric.setId("test_application_1234");
    metric.setKey("test_score");
    metric.setValue((float) 0.666667);
    metric.setWorkerIndex("test_worker-2");
    metric.setTimestamp(timestamp);
    metric.setStep(0);
    metric.setIsNan(false);
    boolean result = metricService.insert(metric);
    assertTrue(result);

    Metric metricDb2 = metricService.selectByPrimaryKeySelective(metric).get(0);
    boolean deleteResult = metricService.deleteById(metricDb2.getId());
    assertTrue(deleteResult);
  }

  private void compareMetrics(Metric metric, Metric metricDb) {
    assertEquals(metric.getId(), metricDb.getId());
    assertEquals(metric.getId(), metricDb.getId());
    assertEquals(metric.getIsNan(), metricDb.getIsNan());
    assertEquals(metric.getKey(), metricDb.getKey());
    assertEquals(metric.getStep(), metricDb.getStep());
    assertEquals(metric.getTimestamp(), metricDb.getTimestamp());
    assertEquals(metric.getValue(), metricDb.getValue());
    assertEquals(metric.getWorkerIndex(), metricDb.getWorkerIndex());
  }
}
