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

//import java.math.BigInteger;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class ParamServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(ParamServiceTest.class);
  ParamService paramService = new ParamService();

  @After
  public void removeAllRecord() throws Exception {
    List<Param> paramList = paramService.selectAll();
    LOG.info("jobList.size():{}", paramList.size());
    for (Param param : paramList) {
      paramService.deleteById(param.getId());
    }
  }

  /*
# +----------+-------+--------------+-----------------------+
# | key      | value | worker_index | job_name              |
# +----------+-------+--------------+-----------------------+
# | max_iter | 100   | worker-1     | application_123651651 |
# | n_jobs   | 5     | worker-1     | application_123456898 |
# | alpha    | 20    | worker-1     | application_123456789 |
# +----------+-------+--------------+-----------------------+
  */

  @Test
  public void testSelectParam() throws Exception {
    Param param = new Param();
    param.setParam_key("score");
    param.setValue("199");
    param.setWorker_index("worker-1");
    param.setJob_name("application_123651651");
    param.setCreateBy("ParamServiceTest-CreateBy");
    boolean result = paramService.insert(param);
    assertNotEquals(result, -1);
    List<Param> paramList = paramService.selectAll();

    assertEquals(paramList.size(), 1);

    Param paramDb = paramList.get(0);
    compareParams(param, paramDb);

    Param paramDb2 = paramService.selectById("" + result);
    compareParams(param, paramDb2);
  }

  @Test
  public void testUpdateJob() throws Exception {
    Param param = new Param();
    param.setParam_key("score");
    param.setValue("100");
    param.setWorker_index("worker-2");
    param.setJob_name("application_1234");
    param.setCreateBy("ParamServiceTest-CreateBy");
    boolean result = paramService.insert(param);
    assertTrue(result);

    param.setParam_key("scoreNew");
    param.setValue("100");
    param.setWorker_index("worker-New");
    param.setJob_name("application_1234New");
    param.setUpdateBy("ParamServiceTest-UpdateBy");
    boolean editResult = paramService.update(param);
    assertTrue(editResult);

    Param paramDb2 = paramService.selectById("" + result);
    compareParams(param, paramDb2);
  }

  @Test
  public void delete() throws Exception {
    Param param = new Param();
    param.setParam_key("score");
    param.setValue("100");
    param.setWorker_index("worker-2");
    param.setJob_name("application_1234");
    param.setCreateBy("ParamServiceTest-CreateBy");

    boolean result = paramService.insert(param);
    assertTrue(result);

    boolean deleteResult = paramService.deleteById("" + result);
    assertTrue(deleteResult);
  }

  private void compareParams(Param param, Param paramDb) {
    assertEquals(param.getId(), paramDb.getId());
    assertEquals(param.getJob_name(), paramDb.getJob_name());
    assertEquals(param.getParam_key(), paramDb.getParam_key());
    assertEquals(param.getValue(), paramDb.getValue());
    assertEquals(param.getWorker_index(), paramDb.getWorker_index());
  }
}
