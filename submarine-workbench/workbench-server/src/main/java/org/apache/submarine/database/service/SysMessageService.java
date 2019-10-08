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
package org.apache.submarine.database.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.SysMessage;
import org.apache.submarine.database.mappers.SysMessageMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SysMessageService {
  private static final Logger LOG = LoggerFactory.getLogger(SysMessageService.class);

  public void add(SysMessage sysMessage) throws Exception {
    LOG.info("add({})", sysMessage.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysMessageMapper sysMessageMapper = sqlSession.getMapper(SysMessageMapper.class);
      sysMessageMapper.insert(sysMessage);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
  }
}
