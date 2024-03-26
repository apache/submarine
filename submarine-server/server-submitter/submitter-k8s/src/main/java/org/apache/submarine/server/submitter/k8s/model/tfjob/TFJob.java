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

/**
 * It's the tf-operator's entry model.
 */
public class TFJob extends MLJob {

  public static final  String CRD_TF_KIND_V1 = "TFJob";
  public static final  String CRD_TF_PLURAL_V1 = "tfjobs";
  public static final  String CRD_TF_GROUP_V1 = "kubeflow.org";
  public static final  String CRD_TF_VERSION_V1 = "v1";
  public static final  String CRD_TF_API_VERSION_V1 = CRD_TF_GROUP_V1 +
      "/" + CRD_TF_VERSION_V1;

  @SerializedName("spec")
  private TFJobSpec spec;

  public TFJob(ExperimentSpec experimentSpec) throws InvalidSpecException {
    super(experimentSpec);
    setApiVersion(CRD_TF_API_VERSION_V1);
    setKind(CRD_TF_KIND_V1);
    setPlural(CRD_TF_PLURAL_V1);
    setVersion(CRD_TF_VERSION_V1);
    setGroup(CRD_TF_GROUP_V1);
    // set spec
    setSpec(parseTFJobSpec(experimentSpec));

    V1Container initContainer = this.getExperimentHandlerContainer(experimentSpec);
    if (initContainer != null) {
      Map<TFJobReplicaType, MLJobReplicaSpec> replicaSet = this.getSpec().getReplicaSpecs();
      if (replicaSet.keySet().contains(TFJobReplicaType.Ps)) {
        MLJobReplicaSpec psSpec = replicaSet.get(TFJobReplicaType.Ps);
        psSpec.getTemplate().getSpec().addInitContainersItem(initContainer);
      } else {
        throw new InvalidSpecException("PreHandler only support TFJob with PS for now");
      }
    }
  }

  @Override
  public CustomResourceType getResourceType() {
    return CustomResourceType.TFJob;
  }

  /**
   * Parse TFJob Spec
   */
  private TFJobSpec parseTFJobSpec(ExperimentSpec experimentSpec)
          throws InvalidSpecException {
    TFJobSpec tfJobSpec = new TFJobSpec();
    Map<TFJobReplicaType, MLJobReplicaSpec> replicaSpecMap = new HashMap<>();

    for (Map.Entry<String, ExperimentTaskSpec> entry : experimentSpec.getSpec().entrySet()) {
      String replicaType = entry.getKey();
      ExperimentTaskSpec taskSpec = entry.getValue();

      if (TFJobReplicaType.isSupportedReplicaType(replicaType)) {
        MLJobReplicaSpec replicaSpec = new MLJobReplicaSpec();
        replicaSpec.setReplicas(taskSpec.getReplicas());
        V1PodTemplateSpec podTemplateSpec = ExperimentSpecParser.parseTemplateSpec(taskSpec, experimentSpec);
        replicaSpec.setTemplate(podTemplateSpec);
        replicaSpecMap.put(TFJobReplicaType.valueOf(replicaType), replicaSpec);
      } else {
        throw new InvalidSpecException("Unrecognized replica type name: " +
            entry.getKey() +
            ", it should be " +
            String.join(",", TFJobReplicaType.names()) +
            " for TensorFlow experiment.");
      }
    }
    tfJobSpec.setReplicaSpecs(replicaSpecMap);
    return tfJobSpec;
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

  @Override
  public Experiment read(K8sClient api) {
    try {
      TFJob tfJob = api.getTfJobClient().get(getMetadata().getNamespace(), getMetadata().getName())
          .throwsApiException().getObject();
      if (LOG.isDebugEnabled()) {
        LOG.debug("Get TFJob resource: \n{}", YamlUtils.toPrettyYaml(tfJob));
      }
      return parseExperimentResponseObject(tfJob, TFJob.class);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Experiment create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create TFJob resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      TFJob tfJob = api.getTfJobClient().create(getMetadata().getNamespace(), this,
          new CreateOptions()).throwsApiException().getObject();
      if (LOG.isDebugEnabled()) {
        LOG.debug("Get TFJob resource: \n{}", YamlUtils.toPrettyYaml(tfJob));
      }
      return parseExperimentResponseObject(tfJob, TFJob.class);
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse TFJob object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse TFJob object failed by " +
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
        LOG.debug("Patch TFJob resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      TFJob tfJob = api.getTfJobClient()
          .patch(getMetadata().getNamespace(), getMetadata().getName(),
              V1Patch.PATCH_FORMAT_APPLY_YAML,
              new V1Patch(JsonUtils.toJson(this)), patchOptions)
          .throwsApiException().getObject();
      return parseExperimentResponseObject(tfJob, TFJob.class);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Experiment delete(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete TFJob resource in namespace: {} and name: {}",
                this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      V1Status status = api.getTfJobClient()
          .delete(getMetadata().getNamespace(), getMetadata().getName(),
              MLJobConverter.toDeleteOptionsFromMLJob(this))
          .getStatus();
      return parseExperimentResponseStatus(status);
    } catch (Exception e) {
      throw new SubmarineRuntimeException(500, e.getMessage());
    }
  }

}
