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

public class PyTorchJob extends MLJob {

  public static final  String CRD_PYTORCH_KIND_V1 = "PyTorchJob";
  public static final  String CRD_PYTORCH_PLURAL_V1 = "pytorchjobs";
  public static final  String CRD_PYTORCH_GROUP_V1 = "kubeflow.org";
  public static final  String CRD_PYTORCH_VERSION_V1 = "v1";
  public static final  String CRD_PYTORCH_API_VERSION_V1 = CRD_PYTORCH_GROUP_V1 +
      "/" + CRD_PYTORCH_VERSION_V1;

  @SerializedName("spec")
  private PyTorchJobSpec spec;

  public PyTorchJob(ExperimentSpec experimentSpec) throws InvalidSpecException {
    super(experimentSpec);
    setApiVersion(CRD_PYTORCH_API_VERSION_V1);
    setKind(CRD_PYTORCH_KIND_V1);
    setPlural(CRD_PYTORCH_PLURAL_V1);
    setVersion(CRD_PYTORCH_VERSION_V1);
    setGroup(CRD_PYTORCH_GROUP_V1);
    // set spec
    setSpec(parsePyTorchJobSpec(experimentSpec));
  }

  @Override
  public CustomResourceType getResourceType() {
    return CustomResourceType.PyTorchJob;
  }

  /**
   * Parse PyTorchJob Spec
   */
  private PyTorchJobSpec parsePyTorchJobSpec(ExperimentSpec experimentSpec)
          throws InvalidSpecException {
    PyTorchJobSpec pyTorchJobSpec = new PyTorchJobSpec();

    V1Container initContainer = this.getExperimentHandlerContainer(experimentSpec);
    Map<PyTorchJobReplicaType, MLJobReplicaSpec> replicaSpecMap = new HashMap<>();
    for (Map.Entry<String, ExperimentTaskSpec> entry : experimentSpec.getSpec().entrySet()) {
      String replicaType = entry.getKey();
      ExperimentTaskSpec taskSpec = entry.getValue();
      if (PyTorchJobReplicaType.isSupportedReplicaType(replicaType)) {
        MLJobReplicaSpec replicaSpec = new MLJobReplicaSpec();
        replicaSpec.setReplicas(taskSpec.getReplicas());
        V1PodTemplateSpec podTemplateSpec = ExperimentSpecParser.parseTemplateSpec(taskSpec, experimentSpec);

        if (initContainer != null && replicaType.equals("Master")) {
          podTemplateSpec.getSpec().addInitContainersItem(initContainer);
        }

        replicaSpec.setTemplate(podTemplateSpec);
        replicaSpecMap.put(PyTorchJobReplicaType.valueOf(replicaType), replicaSpec);
      } else {
        throw new InvalidSpecException("Unrecognized replica type name: " +
            entry.getKey() + ", it should be " +
            String.join(",", PyTorchJobReplicaType.names()) +
            " for PyTorch experiment.");
      }
    }
    pyTorchJobSpec.setReplicaSpecs(replicaSpecMap);
    return pyTorchJobSpec;
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

  @Override
  public Experiment read(K8sClient api) {
    try {
      PyTorchJob pyTorchJob = api.getPyTorchJobClient()
          .get(getMetadata().getNamespace(), getMetadata().getName())
          .throwsApiException().getObject();
      if (LOG.isDebugEnabled()) {
        LOG.debug("Get PyTorchJob resource: \n{}", YamlUtils.toPrettyYaml(pyTorchJob));
      }
      return parseExperimentResponseObject(pyTorchJob, PyTorchJob.class);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Experiment create(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Create PyTorchJob resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      PyTorchJob pyTorchJob = api.getPyTorchJobClient()
          .create(getMetadata().getNamespace(), this, new CreateOptions())
          .throwsApiException().getObject();
      return parseExperimentResponseObject(pyTorchJob, PyTorchJob.class);
    } catch (ApiException e) {
      LOG.error("K8s submitter: parse PyTorchJob object failed by " + e.getMessage(), e);
      throw new SubmarineRuntimeException(e.getCode(), "K8s submitter: parse PyTorchJob object failed by " +
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
        LOG.debug("Patch PyTorchJob resource: \n{}", YamlUtils.toPrettyYaml(this));
      }
      PyTorchJob pyTorchJob = api.getPyTorchJobClient()
          .patch(getMetadata().getNamespace(), getMetadata().getName(),
              V1Patch.PATCH_FORMAT_APPLY_YAML,
              new V1Patch(JsonUtils.toJson(this)),
              patchOptions)
          .throwsApiException().getObject();
      return parseExperimentResponseObject(pyTorchJob, PyTorchJob.class);
    }  catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public Experiment delete(K8sClient api) {
    try {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Delete PyTorchJob resource in namespace: {} and name: {}",
                this.getMetadata().getNamespace(), this.getMetadata().getName());
      }
      V1Status status = api.getPyTorchJobClient()
          .delete(getMetadata().getNamespace(), getMetadata().getName(),
              MLJobConverter.toDeleteOptionsFromMLJob(this))
          .throwsApiException().getStatus();
      return parseExperimentResponseStatus(status);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }
}
