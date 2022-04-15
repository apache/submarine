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

import org.apache.submarine.server.experiment.database.entity.ExperimentEntity;
import org.apache.submarine.server.experiment.database.service.ExperimentService;
import org.apache.submarine.server.database.workbench.entity.ParamEntity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class ParamServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(ParamServiceTest.class);
  ParamService paramService = new ParamService();
  ExperimentService experimentService = new ExperimentService();

  @Before
  public void createExperiment() throws Exception {
    ExperimentEntity entity = new ExperimentEntity();
    String id = "test_application_12345";
    String spec = "{\"value\": 1}";

    entity.setId(id);
    entity.setExperimentSpec(spec);

    experimentService.insert(entity);
  }
  @After
  public void removeAllRecord() throws Exception {
    List<ParamEntity> paramList = paramService.selectAll();
    LOG.info("paramList.size():{}", paramList.size());
    for (ParamEntity param : paramList) {
      paramService.deleteById(param.getId());
    }

    experimentService.selectAll().forEach(e -> experimentService.delete(e.getId()));
  }

  @Test
  public void testSelect() throws Exception {
    ParamEntity param = new ParamEntity();
    param.setId("test_application_12345");
    param.setKey("test_score");
    param.setValue("199");
    param.setWorkerIndex("test_worker-1");
    boolean result = paramService.insert(param);
    assertTrue(result);
    List<ParamEntity> paramList = paramService.selectAll();

    assertEquals(paramList.size(), 1);

    ParamEntity paramDb = paramList.get(0);
    compareParams(param, paramDb);

    ParamEntity paramDb2 = paramService.selectByPrimaryKeySelective(param).get(0);
    compareParams(param, paramDb2);
  }

  @Test
  public void testUpdate() throws Exception {
    ParamEntity param = new ParamEntity();
    param.setId("test_application_12345");
    param.setKey("test_score");
    param.setValue("100");
    param.setWorkerIndex("test_worker-2");
    boolean result = paramService.insert(param);
    assertTrue(result);

    param.setKey("scoreNew");
    param.setValue("100");
    param.setWorkerIndex("worker-New");
    boolean editResult = paramService.update(param);
    assertTrue(editResult);

    ParamEntity paramDb2 = paramService.selectByPrimaryKeySelective(param).get(0);
    compareParams(param, paramDb2);
  }

  @Test
  public void testDelete() throws Exception {
    ParamEntity param = new ParamEntity();
    param.setId("test_application_12345");
    param.setKey("test_score");
    param.setValue("100");
    param.setWorkerIndex("test_worker-2");

    boolean result = paramService.insert(param);
    assertTrue(result);

    ParamEntity paramDb2 = paramService.selectByPrimaryKeySelective(param).get(0);

    boolean deleteResult = paramService.deleteById(paramDb2.getId());
    assertTrue(deleteResult);
  }

  private void compareParams(ParamEntity param, ParamEntity paramDb) {
    assertEquals(param.getId(), paramDb.getId());
    assertEquals(param.getId(), paramDb.getId());
    assertEquals(param.getKey(), paramDb.getKey());
    assertEquals(param.getValue(), paramDb.getValue());
    assertEquals(param.getWorkerIndex(), paramDb.getWorkerIndex());
  }
}
