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

package org.apache.submarine.jobserver.rest.dao;

/**
 * The machine learning job spec the submarine job server can accept.
 * */
public class MLJobSpec {

  String apiVersion;
  // The engine type the job will use, can be tensorflow or pytorch
  String type;
  // The engine version, not image version/tag. For instance, tensorflow v1.13
  String version;
  /**
   * Advanced property. The RM this job will submit to, k8s or yarn.
   * Could be the path of yarn’s config file or k8s kubeconfig file
   * This should be settled when deploy submarine job server.
   */
  String rmConfig;

  /**
   * Advanced property. The image should be inferred from type and version.
   * The normal user should not set this.
   * */
  String dockerImage;

  // The process aware environment variable
  EnvVaraible[] envVars;

  // The components this cluster will consists.
  Component[] components;

  // The user id who submit job
  String uid;

  /**
   * The queue this job will submitted to.
   * In YARN, we call it queue. In k8s, no such concept.
   * It could be namespace’s name.
   */
  String queue;

  /**
   * The user-specified job name for easy search
   * */
  String name;

  /**
   * This could be file/directory which contains multiple python scripts.
   * We should solve dependencies distribution in k8s or yarn.
   * Or we could build images for each projects before submitting the job
   * */
  String projects;

  /**
   * The command uses to start the job. This is very job-specific.
   * We assume the cmd is the same for all containers in a cluster
   * */
  String cmd;

  public MLJobSpec() {}

  public String getUid() {
    return uid;
  }

  public void setUid(String uid) {
    this.uid = uid;
  }

  public String getQueue() {
    return queue;
  }

  public void setQueue(String queue) {
    this.queue = queue;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public String getProjects() {
    return projects;
  }

  public void setProjects(String projects) {
    this.projects = projects;
  }
  public String getCmd() {
    return cmd;
  }

  public void setCmd(String cmd) {
    this.cmd = cmd;
  }

  public Component[] getComponents() {
    return components;
  }

  public void setComponents(
      Component[] components) {
    this.components = components;
  }
  public String getType() {
    return type;
  }

  public void setType(String type) {
    this.type = type;
  }

  public String getVersion() {
    return version;
  }

  public void setVersion(String version) {
    this.version = version;
  }

  public String getRmConfig() {
    return rmConfig;
  }

  public void setRmConfig(String rmConfig) {
    this.rmConfig = rmConfig;
  }

  public String getDockerImage() {
    return dockerImage;
  }

  public void setDockerImage(String dockerImage) {
    this.dockerImage = dockerImage;
  }

  public EnvVaraible[] getEnvVars() {
    return envVars;
  }

  public void setEnvVars(EnvVaraible[] envVars) {
    this.envVars = envVars;
  }

  public String getApiVersion() {
    return apiVersion;
  }

  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }


}
