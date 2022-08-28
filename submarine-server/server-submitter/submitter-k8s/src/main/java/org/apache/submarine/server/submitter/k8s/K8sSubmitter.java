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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.V1Deployment;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.util.generic.options.DeleteOptions;
import io.kubernetes.client.util.generic.options.ListOptions;

import org.apache.commons.lang3.StringUtils;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.k8s.utils.K8sUtils;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.experiment.MlflowInfo;
import org.apache.submarine.server.api.experiment.TensorboardInfo;
import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.client.K8sDefaultClient;
import org.apache.submarine.server.submitter.k8s.model.AgentPod;
import org.apache.submarine.server.submitter.k8s.model.K8sResource;
import org.apache.submarine.server.submitter.k8s.model.common.Configmap;
import org.apache.submarine.server.submitter.k8s.model.istio.IstioVirtualService;
import org.apache.submarine.server.submitter.k8s.model.common.NullResource;
import org.apache.submarine.server.submitter.k8s.model.common.PersistentVolumeClaim;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJob;
import org.apache.submarine.server.submitter.k8s.model.mljob.MLJobFactory;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.seldon.SeldonDeploymentFactory;
import org.apache.submarine.server.submitter.k8s.model.seldon.SeldonResource;
import org.apache.submarine.server.submitter.k8s.util.NotebookUtils;
import org.apache.submarine.server.submitter.k8s.util.OwnerReferenceUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JobSubmitter for Kubernetes Cluster.
 */
public class K8sSubmitter implements Submitter {

  private static final Logger LOG = LoggerFactory.getLogger(K8sSubmitter.class);

  private static final String TF_JOB_SELECTOR_KEY = "tf-job-name=";
  private static final String PYTORCH_JOB_SELECTOR_KEY = "pytorch-job-name=";
  private static final String XGBoost_JOB_SELECTOR_KEY = "xgboost-job-name=";

  // Add an exception Consumer, handle the problem that delete operation does not have the resource
  public static final Function<ApiException, Object> API_EXCEPTION_404_CONSUMER = e -> {
    if (e.getCode() != 404) {
      LOG.error("When submit resource to k8s get ApiException with code " + e.getCode(), e);
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    } else {
      return null;
    }
  };

  private static final String OVERWRITE_JSON;

  static {
    final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    OVERWRITE_JSON = conf.getString(
            SubmarineConfVars.ConfVars.SUBMARINE_NOTEBOOK_DEFAULT_OVERWRITE_JSON);
  }

  // K8s API client for CRD
  private K8sClient k8sClient;

  public K8sSubmitter() {
  }

  public K8sSubmitter(K8sClient k8sClient) {
    this.k8sClient = k8sClient;
  }

  @Override
  public void initialize(SubmarineConfiguration conf) {
    // move k8s clients init to org.apache.submarine.server.submitter.k8s.K8sClient
    // For the compatibility of the current codes, several variables about api client are exposed temporarily
    if (k8sClient == null) {
      k8sClient = new K8sDefaultClient();
    }
  }

  /**
   * Commit resources with transaction
   * @return committed return objects
   */
  public List<Object> resourceTransaction(K8sResource... resources) {
    Map<K8sResource, Object> commits = new LinkedHashMap<>();
    try {
      for (K8sResource resource : resources) {
        if (resource != null) {
          commits.put(resource, resource.create(k8sClient));
        } else {
          commits.put(new NullResource(), null);
        }
      }
      return new ArrayList<>(commits.values());
    } catch (Exception e) {
      if (!commits.isEmpty()) {
        // Rollback is performed in the reverse order of commits
        List<K8sResource> rollbacks = new ArrayList<>(commits.keySet());
        for (int i = rollbacks.size() - 1; i >= 0; i--) {
          K8sResource rollback = rollbacks.get(i);
          if (!(rollback instanceof NullResource)) {
            LOG.info("Rollback resources {}/{}", rollback.getKind(), rollback.getMetadata().getName());
            try {
              rollbacks.get(i).delete(k8sClient);
            } catch (Exception deleteErr) {
              LOG.error("Failed to delete resource. You may need to delete it manually!", deleteErr);
            }
          }
        }
      }
      throw e;
    }
  }

  /**
   * Delete resources with transaction
   * This is an experimental API, Our main consideration is that:
   * k8s resources transactional deletion cannot handle the rollback of transactions well,
   * so we only guarantee the deletion of primary resource for the time being.
   * We can tolerate the deletion failure of other dependent resources,
   * so as to maximize the availability of the deletion API.
   * @param primary primary resource, Failure of this resource will cause API exceptions
   * @param dependentResources dependent resources
   */
  public <T> T deleteResourcesTransaction(K8sResource<T> primary, K8sResource... dependentResources) {
    T returnResource = primary.delete(k8sClient);
    for (K8sResource dependent : dependentResources) {
      try {
        dependent.delete(k8sClient);
      } catch (Exception e) {
        LOG.warn(String.format("Delete %s/%s failed. %s", dependent.getKind(),
                dependent.getMetadata().getName(), e.getMessage()), e);
        // TODO(cdmikechen): Record the error information into audit service for later tracking
      }
    }
    return returnResource;
  }

  public static V1ObjectMeta createMeta(String namespace, String name) {
    V1ObjectMeta metadata = new V1ObjectMeta();
    metadata.setNamespace(namespace);
    metadata.setName(name);
    metadata.setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
    return metadata;
  }

  private String getServerNamespace() {
    return K8sUtils.getNamespace();
  }

  public static DeleteOptions getDeleteOptions(String apiVersion){
    DeleteOptions deleteOptions = new DeleteOptions();
    deleteOptions.setApiVersion(apiVersion);
    return deleteOptions;
  }

  @Override
  public Experiment createExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    try {
      // MLJob K8s resource object
      MLJob mlJob = MLJobFactory.getMLJob(spec);
      mlJob.getMetadata().setOwnerReferences(OwnerReferenceUtils.getOwnerReference());
      // Agent pod K8s resource object
      AgentPod agentPod = new AgentPod(getServerNamespace(), spec.getMeta().getName(),
          mlJob.getResourceType(), spec.getMeta().getExperimentId());
      // commit resources/CRD with transaction
      List<Object> values = resourceTransaction(mlJob, agentPod);
      return (Experiment) values.get(0);
    } catch (InvalidSpecException e) {
      LOG.error(String.format("K8s submitter: parse %s object failed by %s",
              spec.getMeta().getFramework(), e.getMessage()), e);
      throw new SubmarineRuntimeException(500, e.getMessage());
    }
  }

  @Override
  public Experiment findExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    try {
      // MLJob K8s resource object
      MLJob mlJob = MLJobFactory.getMLJob(spec);
      // Read Experiment
      return mlJob.read(k8sClient);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(400, e.getMessage());
    }
  }

  @Override
  public Experiment patchExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    try {
      // MLJob K8s resource object
      MLJob mlJob = MLJobFactory.getMLJob(spec);
      // Patch Experiment
      return mlJob.replace(k8sClient);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(409, e.getMessage());
    } catch (Error e) {
      throw new SubmarineRuntimeException(500, String.format("Unhandled error: %s", e.getMessage()));
    }
  }

  @Override
  public Experiment deleteExperiment(ExperimentSpec spec) throws SubmarineRuntimeException {
    try {
      // MLJob K8s resource object
      MLJob mlJob = MLJobFactory.getMLJob(spec);
      // Agent pod K8s resource object
      AgentPod agentPod = new AgentPod(getServerNamespace(), spec.getMeta().getName(),
              mlJob.getResourceType(), spec.getMeta().getExperimentId());
      // Delete with transaction
      return deleteResourcesTransaction(mlJob, agentPod);
    } catch (InvalidSpecException e) {
      throw new SubmarineRuntimeException(200, e.getMessage());
    }
  }

  @Override
  public ExperimentLog getExperimentLogName(ExperimentSpec spec, String id) {
    ExperimentLog experimentLog = new ExperimentLog();
    experimentLog.setExperimentId(id);
    try {
      ListOptions listOptions = new ListOptions();
      listOptions.setLabelSelector(getJobLabelSelector(spec));
      final V1PodList podList = k8sClient.getPodClient().list(getServerNamespace(), listOptions)
              .throwsApiException().getObject();
      for (V1Pod pod : podList.getItems()) {
        String podName = pod.getMetadata().getName();
        experimentLog.addPodLog(podName, null);
      }
    } catch (final ApiException e) {
      LOG.error("Error when listing pod for experiment:" + spec.getMeta().getName(), e.getMessage());
    }
    return experimentLog;
  }

  @Override
  public ExperimentLog getExperimentLog(ExperimentSpec spec, String id) {
    ExperimentLog experimentLog = new ExperimentLog();
    experimentLog.setExperimentId(id);
    try {
      ListOptions listOptions = new ListOptions();
      listOptions.setLabelSelector(getJobLabelSelector(spec));
      final V1PodList podList = k8sClient.getPodClient().list(getServerNamespace(), listOptions)
              .throwsApiException().getObject();
      for (V1Pod pod : podList.getItems()) {
        String podName = pod.getMetadata().getName();
        String podLog = k8sClient.getCoreApi().readNamespacedPodLog(
                podName, getServerNamespace(), null, Boolean.FALSE, null,
                Integer.MAX_VALUE, null, Boolean.FALSE,
                Integer.MAX_VALUE, null, Boolean.FALSE);

        experimentLog.addPodLog(podName, podLog);
      }
    } catch (final ApiException e) {
      LOG.error("Error when listing pod for experiment:" + spec.getMeta().getName(), e.getMessage());
    }
    return experimentLog;
  }

  @Override
  public TensorboardInfo getTensorboardInfo() throws SubmarineRuntimeException {
    final String name = "submarine-tensorboard";
    try {
      return new TensorboardInfo(isDeploymentAvailable(name));
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  @Override
  public MlflowInfo getMlflowInfo() throws SubmarineRuntimeException {
    final String name = "submarine-mlflow";
    try {
      return new MlflowInfo(isDeploymentAvailable(name));
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
  }

  private boolean isDeploymentAvailable(String name) throws ApiException{
    V1Deployment deploy = k8sClient.getAppsV1Api()
            .readNamespacedDeploymentStatus(name, getServerNamespace(), "true");
    return deploy == null ? false : Optional.ofNullable(deploy.getStatus().getAvailableReplicas())
        .map(ar -> ar > 0).orElse(false); // at least one replica is running
  }

  @Override
  public Notebook createNotebook(NotebookSpec spec, String notebookId) throws SubmarineRuntimeException {
    // index-3: parse notebook custom resource
    NotebookCR notebookCR = new NotebookCR(spec, notebookId, getServerNamespace());
    final String name = notebookCR.getMetadata().getName();
    final String namespace = notebookCR.getMetadata().getNamespace();

    // index-0: workspace pvc
    PersistentVolumeClaim workspace = new PersistentVolumeClaim(namespace,
            String.format("%s-%s", NotebookUtils.PVC_PREFIX, name), NotebookUtils.STORAGE);
    // index-1: user setting pvc
    PersistentVolumeClaim userset = new PersistentVolumeClaim(namespace,
            String.format("%s-user-%s", NotebookUtils.PVC_PREFIX, name), NotebookUtils.DEFAULT_USER_STORAGE);
    // index-2: overwrite.json configmap
    Configmap overwrite = null;
    if (StringUtils.isNotBlank(OVERWRITE_JSON)) {
      overwrite = new Configmap(namespace, String.format("%s-%s", NotebookUtils.OVERWRITE_PREFIX, name),
              NotebookUtils.DEFAULT_OVERWRITE_FILE_NAME, OVERWRITE_JSON);
    }
    // index-4: agent
    AgentPod agentPod = new AgentPod(namespace, spec.getMeta().getName(),
            CustomResourceType.Notebook, notebookId);
    // index-5: notebook VirtualService custom resource
    IstioVirtualService istioVirtualService = new IstioVirtualService(createMeta(namespace, name));

    // commit resources/CRD with transaction
    List<Object> values = resourceTransaction(workspace, userset, overwrite, notebookCR,
            agentPod, istioVirtualService);
    return (Notebook) values.get(3);
  }

  @Override
  public Notebook findNotebook(NotebookSpec spec, String notebookId) throws SubmarineRuntimeException {
    NotebookCR notebookCR = new NotebookCR(spec, notebookId, getServerNamespace());
    Notebook notebook = notebookCR.read(k8sClient);
    if (notebook.getSpec() == null) {
      notebook.setSpec(spec);
    }
    return notebook;
  }

  @Override
  public Notebook deleteNotebook(NotebookSpec spec, String notebookId) throws SubmarineRuntimeException {
    NotebookCR notebookCR = new NotebookCR(spec, notebookId, getServerNamespace());
    final String name = notebookCR.getMetadata().getName();
    final String namespace = notebookCR.getMetadata().getNamespace();

    // delete crd
    Notebook notebook = notebookCR.delete(k8sClient);

    // delete VirtualService
    new IstioVirtualService(createMeta(namespace, name)).delete(k8sClient);

    // delete pvc
    //  workspace pvc
    new PersistentVolumeClaim(namespace, String.format("%s-%s", NotebookUtils.PVC_PREFIX, name),
            NotebookUtils.STORAGE).delete(k8sClient);
    //  user set pvc
    new PersistentVolumeClaim(namespace, String.format("%s-user-%s", NotebookUtils.PVC_PREFIX, name),
            NotebookUtils.DEFAULT_USER_STORAGE).delete(k8sClient);

    // configmap
    if (StringUtils.isNoneBlank(OVERWRITE_JSON)) {
      new Configmap(namespace, String.format("%s-%s", NotebookUtils.OVERWRITE_PREFIX, name))
              .delete(k8sClient);
    }

    // delete agent
    AgentPod agentPod = new AgentPod(namespace, spec.getMeta().getName(),
            CustomResourceType.Notebook, notebookId);
    LOG.info(String.format("Notebook:%s had been deleted, start to delete agent pod:%s",
            spec.getMeta().getName(), agentPod.getMetadata().getName()));
    new AgentPod(namespace, spec.getMeta().getName(), CustomResourceType.Notebook, notebookId)
            .delete(k8sClient);

    return notebook;
  }

  @Override
  public List<Notebook> listNotebook(String id) throws SubmarineRuntimeException {
    List<Notebook> notebookList;
    String namespace = getServerNamespace();

    try {
      ListOptions listOptions = new ListOptions();
      listOptions.setLabelSelector(NotebookCR.NOTEBOOK_OWNER_SELECTOR_KEY + "=" + id);
      Object object = k8sClient.getNotebookCRClient().list(namespace, listOptions)
              .throwsApiException().getObject();
      notebookList = NotebookUtils.parseObjectForList(object);
    } catch (ApiException e) {
      throw new SubmarineRuntimeException(e.getCode(), e.getMessage());
    }
    return notebookList;
  }

  @Override
  public void createServe(ServeSpec spec) throws SubmarineRuntimeException {
    // Seldon Deployment Resource
    SeldonResource seldonDeployment = SeldonDeploymentFactory.getSeldonDeployment(spec);
    // VirtualService Resource
    IstioVirtualService istioVirtualService = seldonDeployment.getIstioVirtualService();
    // commit SeldonResource and IstioVirtualService with transaction
    resourceTransaction(seldonDeployment, istioVirtualService);
  }

  @Override
  public void deleteServe(ServeSpec spec) throws SubmarineRuntimeException {
    // Seldon Deployment Resource
    SeldonResource seldonDeployment = SeldonDeploymentFactory.getSeldonDeployment(spec);
    // VirtualService Resource
    IstioVirtualService istioVirtualService = seldonDeployment.getIstioVirtualService();
    // Delete SeldonResource and IstioVirtualService with transaction
    deleteResourcesTransaction(seldonDeployment, istioVirtualService);
  }

  private String getJobLabelSelector(ExperimentSpec experimentSpec) {
    if (experimentSpec.getMeta().getFramework()
            .equalsIgnoreCase(ExperimentMeta.SupportedMLFramework.TENSORFLOW.getName())) {
      return TF_JOB_SELECTOR_KEY + experimentSpec.getMeta().getExperimentId();
    } else if (experimentSpec.getMeta().getFramework()
            .equalsIgnoreCase(ExperimentMeta.SupportedMLFramework.XGBOOST.getName())) {
      return XGBoost_JOB_SELECTOR_KEY + experimentSpec.getMeta().getExperimentId();
    }
    else {
      return PYTORCH_JOB_SELECTOR_KEY + experimentSpec.getMeta().getExperimentId();
    }
  }

}
