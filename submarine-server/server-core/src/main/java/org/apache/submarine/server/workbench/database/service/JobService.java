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

import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.workbench.database.entity.JobEntity;
import org.apache.submarine.server.workbench.database.mappers.JobMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JobService {
  private static final Logger LOG = LoggerFactory.getLogger(JobService.class);

  public List<JobEntity> queryJobList(
          String userName,
          String column,
          String order,
          int pageNo,
          int pageSize)
          throws Exception {
    LOG.info("queryJobList owner:{}, column:{}, order:{}, pageNo:{}, pageSize:{}",
            userName, column, order, pageNo, pageSize);

    List<JobEntity> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper projectMapper = sqlSession.getMapper(JobMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("userName", userName);
      where.put("column", column);
      where.put("order", order);
      list = projectMapper.selectAll(where, new RowBounds(pageNo, pageSize));
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

  public JobEntity selectByJobId(String jobId) throws Exception {
    LOG.info("select a job by jobid {}", jobId);
    JobEntity job;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper projectMapper = sqlSession.getMapper(JobMapper.class);
      job = projectMapper.selectByJobId(jobId);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return job;
  }

  public JobEntity selectById(String id) throws Exception {
    LOG.info("select a job by id {}", id);
    JobEntity job;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper projectMapper = sqlSession.getMapper(JobMapper.class);
      job = projectMapper.selectByPrimaryKey(id);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return job;
  }

  public boolean add(JobEntity job) throws Exception {
    LOG.info("add({})", job.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper jobMapper = sqlSession.getMapper(JobMapper.class);
      jobMapper.insert(job);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean updateByPrimaryKeySelective(JobEntity job) throws Exception {
    LOG.info("updateByPrimaryKeySelective({})", job.toString());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper jobMapper = sqlSession.getMapper(JobMapper.class);
      jobMapper.updateByPrimaryKeySelective(job);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean delete(String id) throws Exception {
    LOG.info("delete jobs by id {}", id);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper jobMapper = sqlSession.getMapper(JobMapper.class);
      jobMapper.deleteByPrimaryKey(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean deleteByJobId(String jobId) throws Exception {
    LOG.info("delete jobs by jobId {}", jobId);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      JobMapper jobMapper = sqlSession.getMapper(JobMapper.class);
      jobMapper.deleteByJobId(jobId);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

}
