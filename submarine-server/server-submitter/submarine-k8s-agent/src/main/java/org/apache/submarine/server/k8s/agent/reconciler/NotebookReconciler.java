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
import io.javaoperatorsdk.operator.api.reconciler.Context;
import io.javaoperatorsdk.operator.api.reconciler.ControllerConfiguration;
import io.javaoperatorsdk.operator.api.reconciler.Reconciler;
import io.javaoperatorsdk.operator.api.reconciler.UpdateControl;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.database.notebook.mappers.NotebookMapper;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.k8s.agent.model.notebook.NotebookResource;
import org.apache.submarine.server.k8s.agent.model.notebook.status.NotebookCondition;
import org.apache.submarine.server.k8s.utils.OwnerReferenceConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.ZonedDateTime;
import java.util.Date;
import java.util.List;
import java.util.Objects;

import static org.apache.submarine.server.k8s.agent.SubmarineAgentListener.DTF;

/**
 * Notebook Reconciler
 * <p>
 * Submarine will add `notebook-id` and `notebook-owner-id` labels when creating the notebook,
 * so we need to do the filtering.
 * <p>
 * Label selectors reference:
 * https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#api
 */
@ControllerConfiguration(
    labelSelector = "notebook-id,notebook-owner-id",
    generationAwareEventProcessing = false
)
public class NotebookReconciler implements Reconciler<NotebookResource> {

  private static final Logger LOGGER = LoggerFactory.getLogger(NotebookReconciler.class);

  /* reasons */
  private static final String CREATING_REASON = "The notebook instance is creating";
  private static final String RUNNING_REASON = "The notebook instance is running";
  private static final String FAILED_REASON = "The notebook instance is failed";
  private static final String TERMINATING_REASON = "The notebook instance is terminating";

  @Override
  public UpdateControl<NotebookResource> reconcile(NotebookResource notebook,
                                                   Context<NotebookResource> context) {
    LOGGER.debug("Reconciling Notebook: {}", notebook);
    if (!notebook.hasOwnerReferenceFor(OwnerReferenceConfig.getSubmarineUid())) {
      LOGGER.trace("OwnerReference is {}, Skip the processing of this notebook",
          notebook.getMetadata().getOwnerReferences().stream()
              .map(OwnerReference::getUid).findFirst().orElse(null));
    } else {
      triggerStatus(notebook);
    }
    return UpdateControl.noUpdate();
  }

  /**
   * Trigger status
   */
  private void triggerStatus(NotebookResource notebook) {
    if (notebook.getStatus() == null) return;
    List<NotebookCondition> conditions = notebook.getStatus().getConditions();
    // find notebook name/notebook_id
    String name = notebook.getMetadata().getName();
    if (conditions == null || conditions.isEmpty()) {
      LOGGER.warn("{} conditions is empty, skip ...", name);
    } else {
      /*
       * get conditions and update notebook, Here is an example yaml of a state
       * status:
       *   conditions:
       *   - lastProbeTime: "2022-11-24T01:07:12Z"
       *     type: Running
       *   - lastProbeTime: "2022-11-24T01:07:07Z"
       *     message: Error
       *     reason: Error
       *     type: Terminated
       *   - lastProbeTime: "2022-11-23T10:24:57Z"
       *     type: Running
       *   - lastProbeTime: "2022-11-23T10:24:36Z"
       *     reason: PodInitializing
       *     type: Waiting
       *   containerState:
       *     running:
       *       startedAt: "2022-11-24T01:07:00Z"
       *   readyReplicas: 1
       */
      // get sorted latest status
      // Sometimes the status will be out of order after the notebook-controller restarts
      NotebookCondition lastCondition = conditions.stream()
          .max((c1, c2) -> getLastProbeTime(c1).compareTo(getLastProbeTime(c2))).get();
      // The type value can refer to
      // https://github.com/kubeflow/kubeflow/blob/master/components/notebook-controller/api/v1/notebook_types.go#L48
      // Possible values are Running|Waiting|Terminated
      String type = Objects.requireNonNull(lastCondition.getType());
      // The reason value can refer to
      // https://github.com/kubeflow/kubeflow/blob/master/components/notebook-controller/api/v1/notebook_types.go#L46
      // it may be optional
      String reason = getReason(lastCondition);
      // time
      Date date = getLastProbeTime(lastCondition);
      LOGGER.info("current type/status/reason of {} is {} / {} / {}",
          name, type, lastCondition.getStatus(), reason);
      String id = notebook.getMetadata().getLabels().get("notebook-id");
      switch (reason) {
        case "Created":
        case "Scheduled":
          updateNotebookStatus(id, Notebook.Status.STATUS_CREATING, CREATING_REASON, date);
          break;
        case "Started":
        case "Pulled":
          updateNotebookStatus(id, Notebook.Status.STATUS_RUNNING, RUNNING_REASON, date);
          break;
        case "BackOff":
        case "Failed":
          updateNotebookStatus(id, Notebook.Status.STATUS_FAILED, FAILED_REASON, date);
          break;
        case "Pulling":
          updateNotebookStatus(id, Notebook.Status.STATUS_PULLING, CREATING_REASON, date);
          break;
        case "Killing":
          updateNotebookStatus(id, Notebook.Status.STATUS_TERMINATING, TERMINATING_REASON, date);
          break;
        default:
          LOGGER.warn("Unprocessed event type: {}, skip it...", type);
          break;
      }
    }
  }

  /**
   * Get condition reason
   */
  private String getReason(NotebookCondition condition) {
    String reason = condition.getReason();
    if (reason == null || reason.isEmpty()) {
      switch (condition.getType()) {
        case "Running":
          reason = "Started";
          break;
        case "Terminated":
          reason = "Killing";
          break;
        default:
          reason = "Waiting";
          break;
      }
    }
    return reason;
  }

  /**
   * Get condition lastProbeTime with date type
   */
  private Date getLastProbeTime(NotebookCondition condition) {
    ZonedDateTime zdt = ZonedDateTime.parse(condition.getLastProbeTime(), DTF);
    return Date.from(zdt.toInstant());
  }

  /**
   * Update notebook status
   */
  private void updateNotebookStatus(String id, Notebook.Status status, String reason, Date updateTime) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      NotebookMapper mapper = sqlSession.getMapper(NotebookMapper.class);
      mapper.updateStatus(id, status.getValue(), reason, updateTime);
      sqlSession.commit();
    } catch (Exception e) {
      LOGGER.error(e.getMessage(), e);
    }
  }
}
