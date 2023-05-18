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
 * Specification of the desired behavior of an experiment.
 */
public class ExperimentSpec {
  private ExperimentMeta meta;
  private EnvironmentSpec environment;
  private Map<String, ExperimentTaskSpec> spec;
  private Map<String, String> experimentHandlerSpec;
  private CodeSpec code;

  public ExperimentSpec() {}

  public ExperimentMeta getMeta() {
    return meta;
  }

  public void setMeta(ExperimentMeta meta) {
    this.meta = meta;
  }

  public EnvironmentSpec getEnvironment() {
    return environment;
  }

  public void setEnvironment(EnvironmentSpec environmentSpec) {
    this.environment = environmentSpec;
  }

  public Map<String, ExperimentTaskSpec> getSpec() {
    return spec;
  }

  public void setSpec(Map<String, ExperimentTaskSpec> spec) {
    this.spec = spec;
  }

  public CodeSpec getCode() {
    return code;
  }

  public void setCode(CodeSpec code) {
    this.code = code;
  }

  public Map<String, String> getExperimentHandlerSpec() {
    return experimentHandlerSpec;
  }

  public void setExperimentHandlerSpec(Map<String, String> experimentHandlerSpec) {
    this.experimentHandlerSpec = experimentHandlerSpec;
  }

  @Override
  public String toString() {
    return "ExperimentSpec{" +
      "meta=" + meta +
      ", environment=" + environment +
      ", spec=" + spec +
      ", code=" + code +
      '}';
  }
}
