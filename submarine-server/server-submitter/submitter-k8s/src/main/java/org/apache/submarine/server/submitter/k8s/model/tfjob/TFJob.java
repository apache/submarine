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

package org.apache.submarine.server.submitter.k8s.model.tfjob;

import com.google.gson.annotations.SerializedName;

import io.kubernetes.client.common.KubernetesObject;

import org.apache.submarine.server.submitter.k8s.model.MLJob;

/**
 * It's the tf-operator's entry model.
 */
public class TFJob extends MLJob implements KubernetesObject {

  public static final  String CRD_TF_KIND_V1 = "TFJob";
  public static final  String CRD_TF_PLURAL_V1 = "tfjobs";
  public static final  String CRD_TF_GROUP_V1 = "kubeflow.org";
  public static final  String CRD_TF_VERSION_V1 = "v1";
  public static final  String CRD_TF_API_VERSION_V1 = CRD_TF_GROUP_V1 +
      "/" + CRD_TF_VERSION_V1;

  @SerializedName("spec")
  private TFJobSpec spec;

  public TFJob() {
    setApiVersion(CRD_TF_API_VERSION_V1);
    setKind(CRD_TF_KIND_V1);
    setPlural(CRD_TF_PLURAL_V1);
    setVersion(CRD_TF_VERSION_V1);
    setGroup(CRD_TF_GROUP_V1);
  }

  /**
   * Get the job spec which contains all the info for TFJob.
   * @return job spec
   */
  public TFJobSpec getSpec() {
    return spec;
  }

  /**
   * Set the spec, the entry of the TFJob
   * @param spec job spec
   */
  public void setSpec(TFJobSpec spec) {
    this.spec = spec;
  }
}
