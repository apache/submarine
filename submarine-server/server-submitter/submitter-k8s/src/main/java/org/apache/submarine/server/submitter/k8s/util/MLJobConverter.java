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

package org.apache.submarine.server.submitter.k8s.util;

import java.util.List;
import io.kubernetes.client.openapi.models.V1DeleteOptions;
import io.kubernetes.client.openapi.models.V1DeleteOptionsBuilder;
import io.kubernetes.client.openapi.models.V1JobCondition;
import io.kubernetes.client.openapi.models.V1JobStatus;
import io.kubernetes.client.openapi.models.V1Status;
import io.kubernetes.client.openapi.models.V1StatusDetails;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.submitter.k8s.model.MLJob;
import org.joda.time.DateTime;

/**
 * Converter for different types.
 * Such as MLJob to Job, V1Status to Job and others.
 */
public class MLJobConverter {
  public static Experiment toJobFromMLJob(MLJob mlJob) {
    Experiment experiment = new Experiment();
    experiment.setUid(mlJob.getMetadata().getUid());
    DateTime dateTime = mlJob.getMetadata().getCreationTimestamp();
    if (dateTime != null) {
      experiment.setAcceptedTime(dateTime.toString());
      experiment.setStatus(Experiment.Status.STATUS_ACCEPTED.getValue());
    }

    V1JobStatus status = mlJob.getStatus();
    if (status != null) {
      dateTime = status.getStartTime();
      if (dateTime != null) {
        experiment.setCreatedTime(dateTime.toString());
        experiment.setStatus(Experiment.Status.STATUS_CREATED.getValue());
      }

      List<V1JobCondition> conditions = status.getConditions();
      if (conditions != null && conditions.size() > 1) {
        experiment.setStatus(Experiment.Status.STATUS_RUNNING.getValue());
        for (V1JobCondition condition : conditions) {
          if (condition.getType().toLowerCase().equals("running")) {
            dateTime = condition.getLastTransitionTime();
            experiment.setRunningTime(dateTime.toString());
            break;
          }
        }
      }

      dateTime = status.getCompletionTime();
      if (conditions != null && dateTime != null) {
        experiment.setFinishedTime(dateTime.toString());
        if ("Succeeded".equalsIgnoreCase(conditions.get(conditions.size() - 1).getType())) {
          experiment.setStatus(Experiment.Status.STATUS_SUCCEEDED.getValue());
        } else if ("Failed".equalsIgnoreCase(conditions.get(conditions.size() - 1).getType())) {
          experiment.setStatus(Experiment.Status.STATUS_FAILED.getValue());
        }
      }
    }
    return experiment;
  }

  public static Experiment toJobFromStatus(V1Status status) {
    Experiment experiment = new Experiment();
    V1StatusDetails details = status.getDetails();
    if (details != null) {
      experiment.setUid(details.getUid());
    }
    if (status.getStatus().toLowerCase().equals("success")) {
      experiment.setStatus(Experiment.Status.STATUS_DELETED.getValue());
    }
    return experiment;
  }

  public static V1DeleteOptions toDeleteOptionsFromMLJob(MLJob job) {
    return new V1DeleteOptionsBuilder().withApiVersion(job.getApiVersion()).build();
  }
}
