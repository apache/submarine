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

package org.apache.submarine.server.api.experiment;

import org.apache.submarine.server.api.spec.ExperimentSpec;

/**
 * The Generic Machine Learning Job in Submarine.
 */
public class Experiment {
  private ExperimentId experimentId;
  private String name;
  private String uid;
  private String status;
  private String acceptedTime;
  private String createdTime;
  private String runningTime;
  private String finishedTime;
  private ExperimentSpec spec;

  /**
   * Get the job id which is unique in submarine
   * @return job id
   */
  public ExperimentId getExperimentId() {
    return experimentId;
  }

  /**
   * Set the job id which generated by submarine
   * @param experimentId job id
   */
  public void setExperimentId(ExperimentId experimentId) {
    this.experimentId = experimentId;
  }

  /**
   * Get the job name which specified by user through the JobSpec
   * @return the job name
   */
  public String getName() {
    return name;
  }

  /**
   * Set the job name which specified by user
   * @param name job name
   */
  public void setName(String name) {
    this.name = name;
  }

  /**
   * The unique identifier for the job, used to retire the job info from the cluster management
   * system.
   *
   * In YARN cluster it best to set the ApplicationId, and in K8s cluster it maybe the job name.
   * @return the unique identifier
   */
  public String getUid() {
    return uid;
  }

  /**
   * Set the job identifier, in YARN cluster it best to set the application id, and in K8s cluster
   * it maybe the uid.
   * @param uid application id (YARN) or uid (K8s)
   */
  public void setUid(String uid) {
    this.uid = uid;
  }

  public String getStatus() {
    return status;
  }

  public void setStatus(String status) {
    this.status = status;
  }

  public String getAcceptedTime() {
    return acceptedTime;
  }

  public void setAcceptedTime(String acceptedTime) {
    this.acceptedTime = acceptedTime;
  }

  public String getCreatedTime() {
    return createdTime;
  }

  public void setCreatedTime(String creatTime) {
    this.createdTime = creatTime;
  }

  public String getRunningTime() {
    return runningTime;
  }

  public void setRunningTime(String runningTime) {
    this.runningTime = runningTime;
  }

  public String getFinishedTime() {
    return finishedTime;
  }

  public void setFinishedTime(String finishedTime) {
    this.finishedTime = finishedTime;
  }

  public ExperimentSpec getSpec() {
    return spec;
  }

  public void setSpec(ExperimentSpec spec) {
    this.spec = spec;
  }

  public enum Status {
    STATUS_ACCEPTED("Accepted"),
    STATUS_CREATED("Created"),
    STATUS_RUNNING("Running"),
    STATUS_SUCCEEDED("Succeeded"),
    STATUS_DELETED("Deleted");

    private String value;
    Status(String value) {
      this.value = value;
    }

    public String getValue() {
      return value;
    }

    @Override
    public String toString() {
      return value;
    }
  }

  public void rebuild(Experiment experiment) {
    if (experiment != null) {
      if (experiment.getExperimentId() != null) {
        this.setExperimentId(experiment.getExperimentId());
      }
      if (experiment.getName() != null) {
        this.setName(experiment.getName());
      }
      if (experiment.getUid() != null) {
        this.setUid(experiment.getUid());
      }
      if (experiment.getSpec() != null) {
        this.setSpec(experiment.getSpec());
      }
      if (experiment.getStatus() != null) {
        this.setStatus(experiment.getStatus());
      }
      if (experiment.getAcceptedTime() != null) {
        this.setAcceptedTime(experiment.getAcceptedTime());
      }
      if (experiment.getCreatedTime() != null) {
        this.setCreatedTime(experiment.getCreatedTime());
      }
      if (experiment.getRunningTime() != null) {
        this.setRunningTime(experiment.getRunningTime());
      }
      if (experiment.getFinishedTime() != null) {
        this.setFinishedTime(experiment.getFinishedTime());
      }
    }
  }
}
