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

public class NotebookSpec {

  private NotebookMeta meta;
  private EnvironmentSpec environment;
  private NotebookPodSpec spec;

  public NotebookSpec() {

  }

  public NotebookMeta getMeta() {
    return meta;
  }

  public void setMeta(NotebookMeta meta) {
    this.meta = meta;
  }

  public EnvironmentSpec getEnvironment() {
    return environment;
  }

  public void setEnvironment(EnvironmentSpec environmentSpec) {
    this.environment = environmentSpec;
  }

  public NotebookPodSpec getSpec() {
    return spec;
  }

  public void setSpec(NotebookPodSpec podSpec) {
    this.spec = podSpec;
  }

}
