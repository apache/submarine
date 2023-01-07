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

package org.apache.submarine.server.database.experiment.mappers;

import org.apache.ibatis.annotations.Param;
import org.apache.submarine.server.database.experiment.entity.ExperimentEntity;

import java.util.Date;
import java.util.List;

public interface ExperimentMapper {

  List<ExperimentEntity> selectAll();

  ExperimentEntity select(String id);

  int insert(ExperimentEntity experiment);

  int update(ExperimentEntity experiment);

  int delete(String id);

  /**
   * Update experimentStatus to 'Created'
   */
  int create(@Param("id") String id, @Param("acceptedTime") Date acceptedTime);

  /**
   * Update experimentStatus to 'Succeeded'
   */
  int succeed(@Param("id") String id, @Param("finishedTime") Date finishedTime);

  /**
   * Update experimentStatus to 'Failed'
   */
  int failed(@Param("id") String id, @Param("finishedTime") Date finishedTime);

  /**
   * Update experimentStatus to 'Running'
   */
  int running(@Param("id") String id, @Param("runningTime") Date runningTime);
}
