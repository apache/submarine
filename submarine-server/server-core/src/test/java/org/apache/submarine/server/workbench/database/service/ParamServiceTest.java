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

import org.apache.submarine.server.workbench.database.entity.Param;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class ParamServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(ParamServiceTest.class);
  ParamService paramService = new ParamService();

  @After
  public void removeAllRecord() throws Exception {
    List<Param> paramList = paramService.selectAll();
    LOG.info("paramList.size():{}", paramList.size());
    for (Param param : paramList) {
      paramService.deleteById(param.getId());
    }
  }

  @Test
  public void testSelect() throws Exception {
    Param param = new Param();
    param.setParamKey("score");
    param.setValue("199");
    param.setWorkerIndex("worker-1");
    param.setJobName("application_123651651");
    boolean result = paramService.insert(param);
    assertTrue(result);
    List<Param> paramList = paramService.selectAll();

    assertEquals(paramList.size(), 1);

    Param paramDb = paramList.get(0);
    compareParams(param, paramDb);
    
    Param paramDb2 = paramService.selectByPrimaryKeySelective(param).get(0);
    compareParams(param, paramDb2);
  }

  @Test
  public void testUpdate() throws Exception {
    Param param = new Param();
    param.setParamKey("score");
    param.setValue("100");
    param.setWorkerIndex("worker-2");
    param.setJobName("application_1234");
    boolean result = paramService.insert(param);
    assertTrue(result);

    param.setParamKey("scoreNew");
    param.setValue("100");
    param.setWorkerIndex("worker-New");
    param.setJobName("application_1234New");
    boolean editResult = paramService.update(param);
    assertTrue(editResult);
    
    Param paramDb2 = paramService.selectByPrimaryKeySelective(param).get(0);
    compareParams(param, paramDb2);
  }

  @Test
  public void testDelete() throws Exception {
    Param param = new Param();
    param.setParamKey("score");
    param.setValue("100");
    param.setWorkerIndex("worker-2");
    param.setJobName("application_1234");

    boolean result = paramService.insert(param);
    assertTrue(result);

    Param paramDb2 = paramService.selectByPrimaryKeySelective(param).get(0);

    boolean deleteResult = paramService.deleteById(paramDb2.getId());
    assertTrue(deleteResult);
  }

  private void compareParams(Param param, Param paramDb) {
    assertEquals(param.getId(), paramDb.getId());
    assertEquals(param.getJobName(), paramDb.getJobName());
    assertEquals(param.getParamKey(), paramDb.getParamKey());
    assertEquals(param.getValue(), paramDb.getValue());
    assertEquals(param.getWorkerIndex(), paramDb.getWorkerIndex());
  }
}
