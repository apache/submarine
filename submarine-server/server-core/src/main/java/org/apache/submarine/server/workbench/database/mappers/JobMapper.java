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
package org.apache.submarine.server.workbench.database.mappers;

import org.apache.ibatis.session.RowBounds;
import org.apache.submarine.server.workbench.database.entity.Job;

import java.util.List;
import java.util.Map;

public interface JobMapper {
  List<Job> selectAll(Map<String, Object> where, RowBounds rowBounds);

  int deleteByPrimaryKey(String id);

  int deleteByJobId(String jobId);

  int insert(Job job);

  int insertSelective(Job job);

  Job selectByPrimaryKey(String id);

  Job selectByJobId(String jobId);

  int updateByPrimaryKeySelective(Job job);

  int updateByPrimaryKey(Job job);
}
