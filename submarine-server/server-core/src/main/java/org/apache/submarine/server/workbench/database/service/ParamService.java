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
//import org.apache.ibatis.session.SqlSession;
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
  

  public int deleteByPrimaryKey(String id) throws Exception {
    int result = -1;
    LOG.info("Metric deleteByPrimaryKey {}", id);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      result = mapper.deleteByPrimaryKey(id);
      sqlSession.commit();

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return result;
  }

  public int insert(Param param) throws Exception {
    int result = -1;
    LOG.info("Param insert {}", param);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      result = mapper.insert(param);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return result;
  }
  
  public Param selectByPrimaryKey(String id) throws Exception {
    LOG.info("Param selectByPrimaryKey {}", id);
    Param param;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      param = mapper.selectByPrimaryKey(id);

    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return param;
  }

  public int update(Param param) throws Exception {
    int result = -1;
    LOG.info("Metric update {}", param);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ParamMapper mapper = sqlSession.getMapper(ParamMapper.class);
      result = mapper.update(param);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return result;
  }
}
