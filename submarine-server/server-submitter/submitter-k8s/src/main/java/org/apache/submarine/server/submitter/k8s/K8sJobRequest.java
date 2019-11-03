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

package org.apache.submarine.server.submitter.k8s;

/**
 * Job request for Kubernetes Submitter.
 */
// TODO: It should implement the JobRequest interface
public class K8sJobRequest {
  private Path path;
  private Object body;
  private String jobName;

  public K8sJobRequest(Path path, Object body) {
    this(path, body, null);
  }

  public K8sJobRequest(Path path, Object body, String jobName) {
    this.path = path;
    this.body = body;
    this.jobName = jobName;
  }

  public void setPath(Path path) {
    this.path = path;
  }

  public Path getPath() {
    return path;
  }

  public void setBody(Object body) {
    this.body = body;
  }

  public Object getBody() {
    return body;
  }

  public String getJobName() {
    return jobName;
  }

  static class Path {
    private String group;
    private String apiVersion;
    private String namespace;
    private String plural;

    Path(String group, String apiVersion, String namespace, String plural) {
      this.group = group;
      this.apiVersion = apiVersion;
      this.namespace = namespace;
      this.plural = plural;
    }

    public String getGroup() {
      return group;
    }

    public String getApiVersion() {
      return apiVersion;
    }

    public String getNamespace() {
      return namespace;
    }

    public String getPlural() {
      return plural;
    }
  }
}
