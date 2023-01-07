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

package org.apache.submarine.server.k8s.agent.reconciler;

import io.fabric8.kubernetes.api.model.OwnerReference;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.database.experiment.mappers.ExperimentMapper;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.k8s.agent.model.training.JobResource;
import org.apache.submarine.server.k8s.agent.model.training.status.JobCondition;
import org.apache.submarine.server.k8s.utils.OwnerReferenceConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.ZonedDateTime;
import java.util.Date;
import java.util.List;
import java.util.Objects;

import static org.apache.submarine.server.k8s.agent.SubmarineAgentListener.DTF;

/**
 * Training Operator CR Job Reconciler
 */
public abstract class JobReconciler<T extends JobResource> {

  public abstract CustomResourceType type();

  private final Logger LOGGER = LoggerFactory.getLogger(getClass());

  /**
   * Update experiment status to 'Created'
   */
  protected void create(String id, Date acceptedTime) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.create(id, acceptedTime);
      sqlSession.commit();
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
    }
  }

  /**
   * Update experiment status to 'Succeeded'
   */
  protected void succeed(String id, Date finishedTime) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.succeed(id, finishedTime);
      sqlSession.commit();
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
    }
  }

  /**
   * Update experiment status to 'Failed'
   */
  protected void failed(String id, Date finishedTime) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.failed(id, finishedTime);
      sqlSession.commit();
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
    }
  }

  /**
   * Update experiment status to 'Running'
   */
  protected void running(String id, Date runningTime) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.running(id, runningTime);
      sqlSession.commit();
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
    }
  }

  /**
   * Trigger status
   */
  protected void triggerStatus(T resource) {
    LOGGER.debug("Reconciling {}: {}", type(), resource);
    if (!resource.hasOwnerReferenceFor(OwnerReferenceConfig.getSubmarineUid())) {
      LOGGER.trace("OwnerReference is {}, Skip the processing of this job",
          resource.getMetadata().getOwnerReferences().stream()
              .map(OwnerReference::getUid).findFirst().orElse(null));
      return;
    }
    /*
     * get conditions, Here is an example yaml of a state
     * status:
     *   completionTime: "2022-11-23T02:23:21Z"
     *   conditions:
     *   - lastTransitionTime: "2022-11-23T02:20:51Z"
     *     lastUpdateTime: "2022-11-23T02:20:51Z"
     *     message: TFJob experiment-1669169951603-0001 is created.
     *     reason: TFJobCreated
     *     status: "True"
     *     type: Created
     *   - lastTransitionTime: "2022-11-23T02:20:52Z"
     *     lastUpdateTime: "2022-11-23T02:20:52Z"
     *     message: TFJob submarine-user-test/experiment-1669169951603-0001 is running.
     *     reason: TFJobRunning
     *     status: "False"
     *     type: Running
     *   - lastTransitionTime: "2022-11-23T02:23:21Z"
     *     lastUpdateTime: "2022-11-23T02:23:21Z"
     *     message: TFJob submarine-user-test/experiment-1669169951603-0001 successfully
     *       completed.
     *     reason: TFJobSucceeded
     *     status: "True"
     *     type: Succeeded
     *   replicaStatuses:
     *     Worker:
     *       succeeded: 2
     *   startTime: "2022-11-23T02:20:51Z"
     */
    if (resource.getStatus() == null) return;
    List<JobCondition> conditions = resource.getStatus().getConditions();
    // find experiment name/experiment_id
    String name = resource.getMetadata().getName();
    if (conditions == null || conditions.isEmpty()) {
      LOGGER.warn("{} conditions is empty, skip ...", name);
    } else {
      // get condition and update experiment
      JobCondition lastCondition = conditions.get(conditions.size() - 1);
      // The reason value can refer to https://github.com/kubeflow/common/blob/master/pkg/util/status.go
      String reason = Objects.requireNonNull(lastCondition.getReason());
      // The type value can refer to https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L112
      String type = Objects.requireNonNull(lastCondition.getType());
      // time
      ZonedDateTime zdt = ZonedDateTime.parse(lastCondition.getLastTransitionTime(), DTF);
      Date date = Date.from(zdt.toInstant());
      LOGGER.info("current type/status/reason of {} is {} / {} / {}",
          name, type, lastCondition.getStatus(), reason);
      switch (type) {
        case "Created":
          create(name, date);
          break;
        case "Restarting":
        case "Running":
          running(name, date);
          break;
        case "Succeeded":
          succeed(name, date);
          break;
        case "Failed":
          failed(name, date);
          break;
        default:
          LOGGER.warn("Unprocessed event type: {}, skip it...", type);
          break;
      }
    }
  }

}
