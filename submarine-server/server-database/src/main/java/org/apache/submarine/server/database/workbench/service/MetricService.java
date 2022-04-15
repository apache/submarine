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

import org.apache.submarine.server.database.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.workbench.entity.MetricEntity;
import org.apache.submarine.server.database.workbench.mappers.MetricMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import org.apache.ibatis.session.SqlSession;

public class MetricService {
  private static final Logger LOG = LoggerFactory.getLogger(MetricService.class);

  public MetricService() {
  }

  public List<MetricEntity> selectAll() throws Exception {
    List<MetricEntity> result;
    LOG.info("Metric selectAll");

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      result = mapper.selectAll();
      sqlSession.commit();

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return result;
  }

  public boolean deleteById(String id) throws Exception {
    LOG.info("Metric deleteByPrimaryKey {}", id);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      mapper.deleteById(id);
      sqlSession.commit();

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean insert(MetricEntity metric) throws Exception {
    LOG.info("Metric insert {}", metric);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      mapper.insert(metric);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public MetricEntity selectById(String id) throws Exception {
    LOG.info("Metric selectByPrimaryKey {}", id);
    MetricEntity metric;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      metric = mapper.selectById(id);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return metric;
  }

  public boolean update(MetricEntity metric) throws Exception {
    LOG.info("Metric update {}", metric);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      mapper.update(metric);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public List<MetricEntity> selectByPrimaryKeySelective(MetricEntity metric) throws Exception {
    List<MetricEntity> result;
    LOG.info("Metric selectByPrimaryKeySelective");

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      MetricMapper mapper = sqlSession.getMapper(MetricMapper.class);
      result = mapper.selectByPrimaryKeySelective(metric);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return result;
  }
}
