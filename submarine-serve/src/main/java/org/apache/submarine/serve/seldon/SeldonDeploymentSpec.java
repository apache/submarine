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

package org.apache.submarine.serve.seldon;

import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;

public class SeldonDeploymentSpec {

  public SeldonDeploymentSpec() {
  }

  public SeldonDeploymentSpec(String protocol) {
    setProtocol(protocol);
  }

  @SerializedName("protocol")
  private String protocol;

  @SerializedName("predictors")
  private List<SeldonPredictor> predictors = new ArrayList<>();

  public String getProtocol() {
    return protocol;
  }

  public void setProtocol(String protocol) {
    this.protocol = protocol;
  }

  public void addPredictor(SeldonPredictor seldonPredictor) {
    predictors.add(seldonPredictor);
  }

  public List<SeldonPredictor> getPredictors() {
    return predictors;
  }

  public void setPredictors(List<SeldonPredictor> predictors) {
    this.predictors = predictors;
  }
}
