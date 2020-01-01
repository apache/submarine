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

package org.apache.submarine.server.api.spec;

import java.util.Map;

/**
 * The submarine job spec for submarine job server. It consist of name, JobLibrarySpec,
 * JobSubmitterSpec, the tasks.
 */
public class JobSpec {
  /**
   * The user-specified job name. Such as: mnist
   */
  private String name;

  /**
   * The user-specified ML framework spec.
   */
  private JobLibrarySpec librarySpec;

  /**
   * The user-specified submitter spec.
   */
  private JobSubmitterSpec submitterSpec;

  /**
   * The tasks of the job.
   * Such as:
   *   TensorFlow: Chief, Ps, Worker, Evaluator
   *   PyTorch: Master, Worker
   */
  private Map<String, JobTaskSpec> taskSpecs;

  public JobSpec() {

  }

  /**
   * Get the job name which specified by user.
   * @return job name
   */
  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  /**
   * Get the library spec.
   * @return JobLibrarySpec
   */
  public JobLibrarySpec getLibrarySpec() {
    return librarySpec;
  }

  public void setLibrarySpec(JobLibrarySpec librarySpec) {
    this.librarySpec = librarySpec;
  }

  /**
   * Get the submitter spec.
   * @return JobSubmitterSpec
   */
  public JobSubmitterSpec getSubmitterSpec() {
    return submitterSpec;
  }

  public void setSubmitterSpec(JobSubmitterSpec submitterSpec) {
    this.submitterSpec = submitterSpec;
  }

  /**
   * Get all tasks spec
   * @return Map of JobTaskSpec
   */
  public Map<String, JobTaskSpec> getTaskSpecs() {
    return taskSpecs;
  }

  public void setTaskSpecs(Map<String, JobTaskSpec> taskSpecs) {
    this.taskSpecs = taskSpecs;
  }

  public boolean validate() {
    return librarySpec != null && librarySpec.validate()
        && submitterSpec != null && submitterSpec.validate()
        && taskSpecs != null;
  }

  /**
   * This could be file/directory which contains multiple python scripts.
   * We should solve dependencies distribution in k8s or yarn.
   * Or we could build images for each projects before submitting the job
   * */
  String projects;

  public String getProjects() {
    return projects;
  }

  public void setProjects(String projects) {
    this.projects = projects;
  }
}
