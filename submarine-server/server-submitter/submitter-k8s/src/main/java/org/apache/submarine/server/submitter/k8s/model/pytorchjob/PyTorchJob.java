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

package org.apache.submarine.server.submitter.k8s.model.pytorchjob;

import com.google.gson.annotations.SerializedName;
import org.apache.submarine.server.submitter.k8s.model.MLJob;

public class PyTorchJob extends MLJob {


  public static final  String CRD_PYTORCH_KIND_V1 = "PyTorchJob";
  public static final  String CRD_PYTORCH_PLURAL_V1 = "pytorchjobs";
  public static final  String CRD_PYTORCH_GROUP_V1 = "kubeflow.org";
  public static final  String CRD_PYTORCH_VERSION_V1 = "v1";
  public static final  String CRD_PYTORCH_API_VERSION_V1 = CRD_PYTORCH_GROUP_V1 +
      "/" + CRD_PYTORCH_VERSION_V1;

  @SerializedName("spec")
  private PyTorchJobSpec spec;

  public PyTorchJob() {
    setApiVersion(CRD_PYTORCH_API_VERSION_V1);
    setKind(CRD_PYTORCH_KIND_V1);
    setPlural(CRD_PYTORCH_PLURAL_V1);
    setVersion(CRD_PYTORCH_VERSION_V1);
    setGroup(CRD_PYTORCH_GROUP_V1);
  }

  /**
   * Get the job spec which contains PyTorchJob JSON CRD.
   *
   * @return job spec
   */
  public PyTorchJobSpec getSpec() {
    return spec;
  }

  /**
   * Set the spec
   *
   * @param spec job spec
   */
  public void setSpec(PyTorchJobSpec spec) {
    this.spec = spec;
  }
}
