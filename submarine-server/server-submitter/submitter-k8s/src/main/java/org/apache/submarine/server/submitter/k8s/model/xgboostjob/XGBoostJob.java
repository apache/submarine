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

package org.apache.submarine.server.submitter.k8s.model.xgboostjob;

import com.google.gson.annotations.SerializedName;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJob;

public class XGBoostJob extends MLJob {

  public static final  String CRD_XGBOOST_KIND_V1 = "XGBoostJob";
  public static final  String CRD_XGBOOST_PLURAL_V1 = "xgboostjobs";
  public static final  String CRD_XGBOOST_GROUP_V1 = "kubeflow.org";
  public static final  String CRD_XGBOOST_VERSION_V1 = "v1";
  public static final  String CRD_XGBOOST_API_VERSION_V1 = CRD_XGBOOST_GROUP_V1 +
      "/" + CRD_XGBOOST_VERSION_V1;

  @SerializedName("spec")
  private XGBoostJobSpec spec;

  public XGBoostJob() {
    setApiVersion(CRD_XGBOOST_API_VERSION_V1);
    setKind(CRD_XGBOOST_KIND_V1);
    setPlural(CRD_XGBOOST_PLURAL_V1);
    setVersion(CRD_XGBOOST_VERSION_V1);
    setGroup(CRD_XGBOOST_GROUP_V1);
  }

  /**
   * Get the job spec which contains all the info for XGBoostJob.
   * @return job spec
   */
  public XGBoostJobSpec getSpec() {
    return spec;
  }

  /**
   * Set the spec, the entry of the XGBoostJob
   * @param spec job spec
   */
  public void setSpec(XGBoostJobSpec spec) {
    this.spec = spec;
  }
}

