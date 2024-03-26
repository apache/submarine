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
import io.kubernetes.client.custom.V1Patch;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;
import io.kubernetes.client.openapi.models.V1Status;
import io.kubernetes.client.util.generic.options.CreateOptions;
import io.kubernetes.client.util.generic.options.PatchOptions;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTaskSpec;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJob;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJobReplicaSpec;
import org.apache.submarine.server.submitter.k8s.parser.ExperimentSpecParser;
import org.apache.submarine.server.submitter.k8s.util.JsonUtils;
import org.apache.submarine.server.submitter.k8s.util.MLJobConverter;
import org.apache.submarine.server.utils.YamlUtils;

import java.util.HashMap;
import java.util.Map;

public class XGBoostJob extends MLJob {

  public static final  String CRD_XGBOOST_KIND_V1 = "XGBoostJob";
  public static final  String CRD_XGBOOST_PLURAL_V1 = "xgboostjobs";
  public static final  String CRD_XGBOOST_GROUP_V1 = "kubeflow.org";
  public static final  String CRD_XGBOOST_VERSION_V1 = "v1";
  public static final  String CRD_XGBOOST_API_VERSION_V1 = CRD_XGBOOST_GROUP_V1 +
      "/" + CRD_XGBOOST_VERSION_V1;

  @SerializedName("spec")
  private XGBoostJobSpec spec;

  public XGBoostJob(ExperimentSpec experimentSpec) {
    super(experimentSpec);
    setApiVersion(CRD_XGBOOST_API_VERSION_V1);
    setKind(CRD_XGBOOST_KIND_V1);
    setPlural(CRD_XGBOOST_PLURAL_V1);
    setVersion(CRD_XGBOOST_VERSION_V1);
    setGroup(CRD_XGBOOST_GROUP_V1);
    // set spec
    setSpec(parseXGBoostJobSpec(experimentSpec));
  }

  private XGBoostJobSpec parseXGBoostJobSpec(ExperimentSpec experimentSpec)
          throws InvalidSpecException {
    XGBoostJobSpec xGBoostJobSpec = new XGBoostJobSpec();

    Map<XGBoostJobReplicaType, MLJobReplicaSpec> replicaSpecMap = new HashMap<>();

    for (Map.Entry<String, ExperimentTaskSpec> entry : experimentSpec.getSpec().entrySet()) {
      String replicaType = entry.getKey();
      ExperimentTaskSpec taskSpec = entry.getValue();
      V1Container initContainer = this.getExperimentHandlerContainer(experimentSpec);
      if (XGBoostJobReplicaType.isSupportedReplicaType(replicaType)) {
        MLJobReplicaSpec replicaSpec = new MLJobReplicaSpec();
        replicaSpec.setReplicas(taskSpec.getReplicas());
        V1PodTemplateSpec podTemplateSpec = ExperimentSpecParser.parseTemplateSpec(taskSpec, experimentSpec);

        if (initContainer != null && replicaType.equals("Master")) {
          podTemplateSpec.getSpec().addInitContainersItem(initContainer);
        }

        replicaSpec.setTemplate(podTemplateSpec);
        replicaSpecMap.put(XGBoostJobReplicaType.valueOf(replicaType), replicaSpec);
      } else {
        throw new InvalidSpecException("Unrecognized replica type name: " +
            entry.getKey() +
            ", it should be " +
            String.join(",", XGBoostJobReplicaType.names()) +
            " for XGBoost experiment.");
      }
    }
    xGBoostJobSpec.setReplicaSpecs(replicaSpecMap);
    return xGBoostJobSpec;
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

  @Override
  public CustomResourceType getResourceType() {
    return CustomResourceType.XGBoost;
  }

  @Override
  public Experiment read(K8sClient api) {
    try {
      XGBoostJob xgBoostJob = api.getXGBoostJobClient()
          .get(getMetadata().getNamespace(), getMetadata().getName())
          .throwsApiException().getObject();
      if (LOG.isDebugEnabled()) {
        LOG.debug("Get XGBoostJob resource: \n{}", YamlUtils.toPrettyYaml(xgBoostJob));
      }
      return parseExperimentResponseObject(xgBoostJob, XGBoostJob.class);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Experiment create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create XGBoostJob resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      XGBoostJob xgBoostJob = api.getXGBoostJobClient()
            .create(getMetadata().getNamespace(), this, new CreateOptions())
            .throwsApiException().getObject();
      return parseExperimentResponseObject(xgBoostJob, XGBoostJob.class);
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse XGBoostJob object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse XGBoostJob object failed by " +
              e.getMessage());
    }
  }

  @Override
  public Experiment replace(K8sClient api) {
    try {
      // Using apply yaml patch, field manager must be set, and it must be forced.
      // https://kubernetes.io/docs/reference/using-api/server-side-apply/#field-management
      PatchOptions patchOptions = new PatchOptions();
      patchOptions.setFieldManager(getExperimentId());
      patchOptions.setForce(true);
      if (LOG.isDebugEnabled()) {
        LOG.debug("Patch XGBoostJob resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      XGBoostJob xgBoostJob = api.getXGBoostJobClient()
          .patch(getMetadata().getNamespace(), getMetadata().getName(),
              V1Patch.PATCH_FORMAT_APPLY_YAML,
              new V1Patch(JsonUtils.toJson(this)),
              patchOptions)
          .throwsApiException().getObject();
      return parseExperimentResponseObject(xgBoostJob, XGBoostJob.class);
    }  catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Experiment delete(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete XGBoostJob resource in namespace: {} and name: {}",
            this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      V1Status status = api.getXGBoostJobClient()
          .delete(getMetadata().getNamespace(), getMetadata().getName(),
              MLJobConverter.toDeleteOptionsFromMLJob(this))
          .getStatus();
      return parseExperimentResponseStatus(status);
    } catch (Exception e) {
      throw new SubmarineRuntimeException(500, e.getMessage());
    }
  }
}

