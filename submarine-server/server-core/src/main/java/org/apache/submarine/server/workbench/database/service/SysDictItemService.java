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

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.rest.workbench.SysDictRestApi;
import org.apache.submarine.server.workbench.database.entity.SysDictItemEntity;
import org.apache.submarine.server.workbench.database.mappers.SysDictItemMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class SysDictItemService {
  private static final Logger LOG = LoggerFactory.getLogger(SysDictRestApi.class);

  public List<SysDictItemEntity> queryDictByCode(String dictCode) {
    List<SysDictItemEntity> dictItems = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysDictItemMapper dictItemMapper = sqlSession.getMapper(SysDictItemMapper.class);
      dictItems = dictItemMapper.queryDictByCode(dictCode);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return null;
    }
    return dictItems;
  }
}
