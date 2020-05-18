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

import org.apache.submarine.server.workbench.database.MyBatisUtil;
import org.apache.submarine.server.workbench.database.entity.Metric;
import org.apache.submarine.server.workbench.database.mappers.MetricMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;

public class MetricService {
  private static final Logger LOG = LoggerFactory.getLogger(MetricService.class);

  public MetricService() {
    
  }

  /*
  List<Metric> selectAll() {

  }

  int deleteByPrimaryKey(String id) {

  }

  int insert(Metric metric) {

  }
*/
  public Metric selectByPrimaryKey(String id) throws Exception {
    LOG.info("select a metric by PrimaryKey {}", id);
    Metric metric;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      // sqlSession.commit();
      metric = mapper.selectByPrimaryKey(id);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return metric;
  }
  /*
  int updateByPrimaryKey(Metric metric) {
    
  }*/
}
