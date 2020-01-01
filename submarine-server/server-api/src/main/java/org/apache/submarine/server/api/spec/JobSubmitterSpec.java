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

/**
 * The spec of the specified JobSubmitter
 */
public class JobSubmitterSpec {
  /**
   * The type of JobSubmitter which will be selected to submit job. Such as: yarn/yarnservice/k8s
   */
  private String type;

  /**
   * The config path of the specified Resource Manager.
   */
  private String configPath;

  /**
   * It known as queue in Apache Hadoop YARN and namespace in Kubernetes.
   */
  private String namespace;

  private String kind;

  private String apiVersion;

  public JobSubmitterSpec() {

  }

  /**
   * Get the submitter type, which used to select the specified submitter to submit this job
   * @return submitter type, such as yarn/yarnservice/k8s
   */
  public String getType() {
    return type;
  }

  /**
   * Set the submitter type, now supports the yarn/yarnservice/k8s.
   * @param type yarn/yarnservice/k8s
   */
  public void setType(String type) {
    this.type = type;
  }

  /**
   * Get the config path to initialize the submitter. In Apache Hadoop YARN cluster it maybe the
   * path of xxx-site.xml and the kube config for Kubernetes.
   * @return config path
   */
  public String getConfigPath() {
    return configPath;
  }

  public void setConfigPath(String configPath) {
    this.configPath = configPath;
  }

  /**
   * Get the namespace to submit the job.
   * It known as queue in Apache Hadoop YARN and namespace in Kubernetes.
   * @return namespace
   */
  public String getNamespace() {
    return namespace;
  }

  public void setNamespace(String namespace) {
    this.namespace = namespace;
  }

  /**
   * (K8s) Get the CRD kind, which will accept the job
   * @return CRD kind, such as: TFJob/PyTorchJob
   */
  public String getKind() {
    return kind;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  /**
   * (K8s) Get the CRD api version
   * @return version, such as: apache.org/submarine/v1
   */
  public String getApiVersion() {
    return apiVersion;
  }

  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public boolean validate() {
    return type != null && namespace != null;
  }
}
