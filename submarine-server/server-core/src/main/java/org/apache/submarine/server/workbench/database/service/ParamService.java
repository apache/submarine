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

import java.util.List;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.workbench.database.MyBatisUtil;
import org.apache.submarine.server.workbench.database.entity.Param;
import org.apache.submarine.server.workbench.database.mappers.ParamMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ParamService {
  
  private static final Logger LOG = LoggerFactory.getLogger(ParamService.class);

  public List<Param> selectAll() throws Exception {
    LOG.info("Param selectAll");
    List<Param> params;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      params = mapper.selectAll();

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return params;
  }
  
  public boolean deleteById(String id) throws Exception {
    LOG.info("Param deleteById {}", id);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      mapper.deleteById(id);
      sqlSession.commit();

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean insert(Param param) throws Exception {
    LOG.info("Param insert {}", param);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      mapper.insert(param);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }
  
  public Param selectById(String id) throws Exception {
    LOG.info("Param selectById {}", id);
    Param param;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      param = mapper.selectById(id);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return param;
  }

  public boolean update(Param param) throws Exception {
    LOG.info("Param update {}", param);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      mapper.update(param);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public List<Param> selectByPrimaryKeySelective(Param param) throws Exception {
    List<Param> result;
    LOG.info("Param selectByPrimaryKeySelective");

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      result = mapper.selectByPrimaryKeySelective(param);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return result;
  }
}
