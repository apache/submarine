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

package org.apache.submarine.server.database.notebook.mappers;

import org.apache.ibatis.annotations.Param;
import org.apache.submarine.server.database.notebook.entity.NotebookEntity;

import java.util.Date;
import java.util.List;

public interface NotebookMapper {
  List<NotebookEntity> selectAll();

  NotebookEntity select(String id);

  int insert(NotebookEntity notebook);

  int update(NotebookEntity notebook);

  int delete(String id);

  /**
   * Update notebook status
   */
  int updateStatus(@Param("id") String id, @Param("status") String status,
                   @Param("reason") String reason, @Param("updateTime")Date updateTime);
}
